#![deny(clippy::all)]
#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]

use std::collections::VecDeque;
use std::io::ErrorKind;
use std::io::{Cursor, Error as IoError, Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex, OnceLock};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;

use rubato::{
  Fft, FixedSync, Indexing, Resampler, WindowFunction,
  audioadapter_buffers::direct::SequentialSliceOfVecs,
};

use symphonia::core::audio::sample::i24;
use symphonia::core::audio::{Audio, GenericAudioBufferRef};
use symphonia::core::codecs::CodecParameters;
use symphonia::core::codecs::audio::well_known::{CODEC_ID_AAC, CODEC_ID_ALAC, CODEC_ID_OPUS};
use symphonia::core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
use symphonia::core::codecs::registry::CodecRegistry;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo};
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::units::Time;
use symphonia_adapter_libopus::OpusDecoder;

mod pcm;

const SAMPLE_MAX: f32 = 32767.0;
const INITIAL_BUFFER_CAPACITY: usize = 1024 * 1024;
const PROBE_THRESHOLD: usize = 1024;
// Minimum buffered compressed bytes before decode() pulls packets from the
// format reader. This is a batching gate, NOT a correctness requirement:
// Symphonia reports WouldBlock when a packet is incomplete and decode()
// simply returns null until more data arrives. A large value (e.g. 1 MiB)
// delays first PCM for live streams (~65s at 128 kbps). The default is small
// enough to be imperceptible on live streams while still batching file
// decodes; pass 0 to initialize() to decode as soon as a packet is available.
const STREAM_DECODE_THRESHOLD: usize = 16 * 1024;
const RESAMPLE_CHUNK_SIZE: usize = 4096;
const MAX_OUTPUT_FRAMES: usize = 4096;

#[napi(object)]
pub struct Mp4AacPacket {
  pub data: Buffer,
  pub start: f64,
  pub end: f64,
}

#[napi(object)]
pub struct Mp4AacTrack {
  pub sample_rate: u32,
  pub channels: u32,
  pub config: Buffer,
  pub packets: Vec<Mp4AacPacket>,
}

type FetchMp4Range = ThreadsafeFunction<(f64, u32), Promise<Buffer>>;

struct Mp4RangeSource {
  fetch: FetchMp4Range,
  length: Option<u64>,
  position: u64,
}

impl Read for Mp4RangeSource {
  fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
    let available = self
      .length
      .map_or(u64::MAX, |length| length.saturating_sub(self.position));
    let length = available.min(output.len() as u64).min(128 * 1024) as u32;
    if length == 0 {
      return Ok(0);
    }
    /* INFO: Complete each read before Symphonia advances its MP4 sample iterator. */
    let bytes = napi::bindgen_prelude::block_on(async {
      self
        .fetch
        .call_async(Ok((self.position as f64, length)))
        .await?
        .await
    })
    .map_err(|error| IoError::other(error.to_string()))?;
    if bytes.len() > length as usize || (self.length.is_some() && bytes.len() != length as usize) {
      return Err(IoError::new(
        ErrorKind::UnexpectedEof,
        "Incomplete MP4 range",
      ));
    }
    output[..bytes.len()].copy_from_slice(&bytes);
    self.position += bytes.len() as u64;
    Ok(bytes.len())
  }
}

impl Seek for Mp4RangeSource {
  fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
    let position = match from {
      SeekFrom::Start(position) => Some(position),
      SeekFrom::Current(offset) => self.position.checked_add_signed(offset),
      SeekFrom::End(offset) => self
        .length
        .and_then(|length| length.checked_add_signed(offset)),
    }
    .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "Invalid MP4 offset"))?;
    self.position = position;
    Ok(position)
  }
}

impl MediaSource for Mp4RangeSource {
  fn is_seekable(&self) -> bool {
    self.length.is_some()
  }
  fn byte_len(&self) -> Option<u64> {
    self.length
  }
}

struct Mp4AudioState {
  reader: Box<dyn FormatReader + 'static>,
  track_id: u32,
  seconds_per_tick: f64,
  alac: Option<SymphoniaDecoder>,
}

#[napi]
pub struct Mp4AudioReader {
  state: Arc<Mutex<Option<Mp4AudioState>>>,
  codec: String,
  sample_rate: Option<u32>,
  channels: Option<u32>,
  config: Vec<u8>,
}

/// Opens an MP4 using a random-access callback without loading its media data.
#[napi]
pub async fn open_mp4_audio(
  length: Option<f64>,
  #[napi(
    ts_arg_type = "(error: Error | null, range: [start: number, length: number]) => Promise<Buffer>"
  )]
  fetch: FetchMp4Range,
  seek_seconds: Option<f64>,
) -> Result<Mp4AudioReader> {
  if length.is_some_and(|length| {
    !(1.0..=9_007_199_254_740_991.0).contains(&length) || length.fract() != 0.0
  }) {
    return Err(Error::from_reason("Invalid MP4 byte length"));
  }
  spawn_blocking(move || {
    let source = MediaSourceStream::new(
      Box::new(Mp4RangeSource {
        fetch,
        length: length.map(|length| length as u64),
        position: 0,
      }),
      Default::default(),
    );
    let mut reader = probe_mp4(source)?;
    let (track, parameters, codec) = reader
      .tracks()
      .iter()
      .find_map(|track| {
        let parameters = track.codec_params.as_ref()?.audio()?;
        let codec = match parameters.codec {
          CODEC_ID_AAC => "aac",
          CODEC_ID_ALAC => "alac",
          CODEC_ID_OPUS => "opus",
          _ => return None,
        };
        Some((track, parameters, codec))
      })
      .ok_or_else(|| Error::from_reason("No supported track found in MP4"))?;
    let track_id = track.id;
    let seconds_per_tick = f64::from(
      track
        .time_base
        .ok_or_else(|| Error::from_reason("MP4 audio track has no time base"))?,
    );
    let sample_rate = parameters.sample_rate;
    let channels = parameters
      .channels
      .as_ref()
      .map(|layout| layout.count() as u32);
    let config = parameters
      .extra_data
      .as_deref()
      .unwrap_or_default()
      .to_vec();
    let alac = if codec == "alac" {
      let mut decoder = SymphoniaDecoder::new();
      decoder.configure_audio_decoder(track_id, parameters)?;
      Some(decoder)
    } else {
      None
    };
    if let Some(seconds) = seek_seconds.filter(|seconds| *seconds != 0.0) {
      let time = Time::try_from_secs_f64(seconds)
        .ok_or_else(|| Error::from_reason("Invalid MP4 seek time"))?;
      reader
        .seek(
          SeekMode::Accurate,
          SeekTo::Time {
            time,
            track_id: Some(track_id),
          },
        )
        .map_err(|error| Error::from_reason(format!("MP4 seek failed: {error}")))?;
    }
    Ok(Mp4AudioReader {
      state: Arc::new(Mutex::new(Some(Mp4AudioState {
        reader,
        track_id,
        seconds_per_tick,
        alac,
      }))),
      codec: codec.to_owned(),
      sample_rate,
      channels,
      config,
    })
  })
  .await
  .map_err(|error| Error::from_reason(error.to_string()))?
}

#[napi]
impl Mp4AudioReader {
  #[napi(getter)]
  pub fn codec(&self) -> String {
    self.codec.clone()
  }

  #[napi(getter)]
  pub fn sample_rate(&self) -> Option<u32> {
    self.sample_rate
  }

  #[napi(getter)]
  pub fn channels(&self) -> Option<u32> {
    self.channels
  }

  #[napi(getter)]
  pub fn config(&self) -> Buffer {
    self.config.clone().into()
  }

  #[napi]
  pub async fn read_packets(&self, limit: u32) -> Result<Vec<Mp4AacPacket>> {
    let state = Arc::clone(&self.state);
    spawn_blocking(move || {
      let mut state = state
        .lock()
        .map_err(|error| Error::from_reason(error.to_string()))?;
      let Some(state) = state.as_mut() else {
        return Ok(Vec::new());
      };
      let mut packets = Vec::new();
      while packets.len() < limit.clamp(1, 256) as usize {
        let Some(packet) = state
          .reader
          .next_packet()
          .map_err(|error| Error::from_reason(format!("MP4 packet extraction failed: {error}")))?
        else {
          break;
        };
        if packet.track_id == state.track_id {
          let ticks = packet.dts.get() as f64;
          packets.push(Mp4AacPacket {
            data: packet.data.into_vec().into(),
            start: ticks * state.seconds_per_tick,
            end: (ticks + packet.dur.get() as f64) * state.seconds_per_tick,
          });
        }
      }
      Ok(packets)
    })
    .await
    .map_err(|error| Error::from_reason(error.to_string()))?
  }

  /// Decodes an ALAC track to 48 kHz stereo PCM using the same resampler as other formats.
  #[napi]
  pub async fn read_pcm(&self) -> Result<Option<Buffer>> {
    let state = Arc::clone(&self.state);
    spawn_blocking(move || {
      let mut state = state
        .lock()
        .map_err(|error| Error::from_reason(error.to_string()))?;
      let Some(state) = state.as_mut() else {
        return Ok(None);
      };
      let decoder = state
        .alac
        .as_mut()
        .ok_or_else(|| Error::from_reason("MP4 PCM extraction requires ALAC"))?;
      Ok(
        decoder
          .decode_from_reader(state.reader.as_mut(), true)?
          .map(|result| result.samples),
      )
    })
    .await
    .map_err(|error| Error::from_reason(error.to_string()))?
  }

  #[napi]
  pub async fn close(&self) -> Result<()> {
    let state = Arc::clone(&self.state);
    spawn_blocking(move || {
      state
        .lock()
        .map_err(|error| Error::from_reason(error.to_string()))?
        .take();
      Ok(())
    })
    .await
    .map_err(|error| Error::from_reason(error.to_string()))?
  }
}

fn probe_mp4(source: MediaSourceStream<'_>) -> Result<Box<dyn FormatReader + '_>> {
  let mut hint = Hint::new();
  hint.with_extension("mp4");
  symphonia::default::get_probe()
    .probe(
      &hint,
      source,
      FormatOptions::default(),
      MetadataOptions::default(),
    )
    .map_err(|error| Error::from_reason(format!("Invalid MP4: {error}")))
}

/// Extracts compressed AAC and decode timestamps from a complete MP4/M4A buffer.
#[napi]
pub fn demux_mp4_aac(input: BufferSlice<'_>) -> Result<Mp4AacTrack> {
  /*
    INFO: Borrow the complete input for this synchronous call. Retain only the
      AAC packets owned by the result, never the original MP4 or its video.
    SOURCES:
      - https://napi.rs/docs/concepts/typed-array#borrowed-types-bufferslice-uint8arrayslice-etc
  */
  let source = MediaSourceStream::new(Box::new(Cursor::new(input.as_ref())), Default::default());
  let mut reader = probe_mp4(source)?;
  let (track, codec) = reader
    .tracks()
    .iter()
    .filter_map(|track| Some((track, track.codec_params.as_ref()?.audio()?)))
    .find(|(_, codec)| codec.codec == CODEC_ID_AAC)
    .ok_or_else(|| Error::from_reason("MP4 has no AAC track"))?;
  let track_id = track.id;
  let seconds_per_tick = f64::from(
    track
      .time_base
      .ok_or_else(|| Error::from_reason("MP4 AAC track has no time base"))?,
  );
  let sample_rate = codec
    .sample_rate
    .ok_or_else(|| Error::from_reason("MP4 AAC track has no sample rate"))?;
  let channels = codec
    .channels
    .as_ref()
    .map(|channels| channels.count() as u32)
    .ok_or_else(|| Error::from_reason("MP4 AAC track has no channel layout"))?;
  let config = codec
    .extra_data
    .as_deref()
    .ok_or_else(|| Error::from_reason("MP4 AAC track has no decoder configuration"))?
    .to_vec()
    .into();
  let mut packets = Vec::new();
  while let Some(packet) = reader
    .next_packet()
    .map_err(|error| Error::from_reason(format!("MP4 packet extraction failed: {error}")))?
  {
    if packet.track_id == track_id {
      let ticks = packet.dts.get() as f64;
      packets.push(Mp4AacPacket {
        data: packet.data.into_vec().into(),
        start: ticks * seconds_per_tick,
        end: (ticks + packet.dur.get() as f64) * seconds_per_tick,
      });
    }
  }
  Ok(Mp4AacTrack {
    sample_rate,
    channels,
    config,
    packets,
  })
}

#[derive(Default)]
struct BufferState {
  data: VecDeque<u8>,
  eos: bool,
  probe_cursor: usize,
  committed: bool,
}

#[derive(Clone)]
struct SharedBuffer {
  state: Arc<Mutex<BufferState>>,
}

impl SharedBuffer {
  fn new() -> Self {
    Self {
      state: Arc::new(Mutex::new(BufferState {
        data: VecDeque::with_capacity(INITIAL_BUFFER_CAPACITY),
        eos: false,
        probe_cursor: 0,
        committed: false,
      })),
    }
  }

  fn push(&self, bytes: &[u8]) {
    self.state.lock().unwrap().data.extend(bytes);
  }

  fn close(&self) {
    self.state.lock().unwrap().eos = true;
  }

  fn clear(&self) {
    let mut state = self.state.lock().unwrap();
    state.data.clear();
    state.eos = false;
    state.probe_cursor = 0;
    state.committed = false;
  }

  fn start_probe(&self) {
    let mut state = self.state.lock().unwrap();
    state.committed = false;
    state.probe_cursor = 0;
  }

  fn reset_probe(&self) {
    self.state.lock().unwrap().probe_cursor = 0;
  }

  fn commit_probe(&self) {
    let mut state = self.state.lock().unwrap();
    if state.committed {
      return;
    }
    let drain = state.probe_cursor.min(state.data.len());
    if drain > 0 {
      state.data.drain(..drain);
    }
    state.probe_cursor = 0;
    state.committed = true;
  }

  fn available_bytes(&self) -> usize {
    let state = self.state.lock().unwrap();
    if state.committed {
      state.data.len()
    } else {
      state.data.len().saturating_sub(state.probe_cursor)
    }
  }

  fn is_closed(&self) -> bool {
    self.state.lock().unwrap().eos
  }
}

struct StreamingSource {
  buffer: SharedBuffer,
}

impl StreamingSource {
  fn copy_from_deque_at(deque: &VecDeque<u8>, start: usize, dst: &mut [u8]) -> usize {
    if start >= deque.len() || dst.is_empty() {
      return 0;
    }

    let (front, back) = deque.as_slices();
    let total = front.len() + back.len();
    let max_copy = dst.len().min(total - start);

    if start < front.len() {
      let front_offset = start;
      let front_available = front.len() - front_offset;
      let front_copy = max_copy.min(front_available);
      dst[..front_copy].copy_from_slice(&front[front_offset..front_offset + front_copy]);

      let remaining = max_copy - front_copy;
      if remaining > 0 {
        dst[front_copy..front_copy + remaining].copy_from_slice(&back[..remaining]);
      }
      max_copy
    } else {
      let back_offset = start - front.len();
      let back_available = back.len().saturating_sub(back_offset);
      let back_copy = max_copy.min(back_available);
      dst[..back_copy].copy_from_slice(&back[back_offset..back_offset + back_copy]);
      back_copy
    }
  }
}

impl Read for StreamingSource {
  fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut state = self.buffer.state.lock().unwrap();

    if state.committed {
      if state.data.is_empty() {
        return if state.eos {
          Ok(0)
        } else {
          Err(IoError::new(ErrorKind::WouldBlock, "need more data"))
        };
      }

      let (front, back) = state.data.as_slices();
      let to_copy = buf.len().min(front.len() + back.len());
      let front_copy = to_copy.min(front.len());

      buf[..front_copy].copy_from_slice(&front[..front_copy]);
      if front_copy < to_copy {
        buf[front_copy..to_copy].copy_from_slice(&back[..to_copy - front_copy]);
      }

      state.data.drain(..to_copy);
      Ok(to_copy)
    } else {
      let available = state.data.len().saturating_sub(state.probe_cursor);
      if available == 0 {
        return if state.eos {
          Ok(0)
        } else {
          Err(IoError::new(ErrorKind::WouldBlock, "need more data"))
        };
      }

      let to_copy = buf.len().min(available);
      let copied = Self::copy_from_deque_at(&state.data, state.probe_cursor, &mut buf[..to_copy]);
      state.probe_cursor += copied;
      Ok(copied)
    }
  }
}

impl Seek for StreamingSource {
  fn seek(&mut self, _: SeekFrom) -> std::io::Result<u64> {
    Err(IoError::new(ErrorKind::Unsupported, "seek not supported"))
  }
}

impl MediaSource for StreamingSource {
  fn is_seekable(&self) -> bool {
    false
  }

  fn byte_len(&self) -> Option<u64> {
    None
  }
}

mod helpers {
  use super::*;

  #[inline]
  pub fn f32_to_i16(s: f32) -> i16 {
    (s.clamp(-1.0, 1.0) * SAMPLE_MAX).round() as i16
  }

  pub fn append_interleaved_from_decoded(
    decoded: &GenericAudioBufferRef<'_>,
    out: &mut Vec<u8>,
    target_channels: usize,
  ) {
    if target_channels == 0 {
      return;
    }
    let frames = decoded.frames();
    out.reserve(frames * target_channels * 2);

    macro_rules! process {
      ($buf:expr, $to_f32:expr) => {{
        let l = $buf.plane(0).unwrap_or(&[]);
        let r = if $buf.num_planes() > 1 {
          $buf.plane(1).unwrap_or(l)
        } else {
          l
        };
        if target_channels == 2 {
          let frames_to_process = frames.min(l.len()).min(r.len());
          let l_slice = &l[..frames_to_process];
          let r_slice = &r[..frames_to_process];
          for (&ls, &rs) in l_slice.iter().zip(r_slice.iter()) {
            out.extend_from_slice(&f32_to_i16($to_f32(ls)).to_le_bytes());
            out.extend_from_slice(&f32_to_i16($to_f32(rs)).to_le_bytes());
          }
        } else {
          for i in 0..frames {
            let lv = l.get(i).map(|&s| $to_f32(s)).unwrap_or(0.0);
            let rv = r.get(i).map(|&s| $to_f32(s)).unwrap_or(0.0);
            if target_channels >= 1 {
              out.extend_from_slice(&f32_to_i16(lv).to_le_bytes());
            }
            if target_channels >= 2 {
              out.extend_from_slice(&f32_to_i16(rv).to_le_bytes());
            }
          }
        }
      }};
    }

    match decoded {
      GenericAudioBufferRef::S16(b) => {
        let l = b.plane(0).unwrap_or(&[]);
        let r = if b.num_planes() > 1 {
          b.plane(1).unwrap_or(l)
        } else {
          l
        };
        if target_channels == 2 {
          let frames_to_process = frames.min(l.len()).min(r.len());
          let l_slice = &l[..frames_to_process];
          let r_slice = &r[..frames_to_process];
          for (&ls, &rs) in l_slice.iter().zip(r_slice.iter()) {
            out.extend_from_slice(&ls.to_le_bytes());
            out.extend_from_slice(&rs.to_le_bytes());
          }
        } else {
          for i in 0..frames {
            let lv = l.get(i).copied().unwrap_or(0);
            if target_channels >= 1 {
              out.extend_from_slice(&lv.to_le_bytes());
            }
            if target_channels >= 2 {
              let rv = r.get(i).copied().unwrap_or(lv);
              out.extend_from_slice(&rv.to_le_bytes());
            }
          }
        }
      }
      GenericAudioBufferRef::F32(b) => process!(b, |s: f32| s),
      GenericAudioBufferRef::F64(b) => process!(b, |s: f64| s as f32),
      GenericAudioBufferRef::U8(b) => process!(b, |s: u8| (s as f32 - 128.0) / 128.0),
      GenericAudioBufferRef::S24(b) => {
        process!(b, |s: i24| s.inner() as f32 / 8_388_607.0)
      }
      GenericAudioBufferRef::S32(b) => process!(b, |s: i32| s as f32 / 2_147_483_647.0),
      _ => {}
    }
  }

  pub fn push_decoded_as_planar(
    decoded: &GenericAudioBufferRef<'_>,
    accum: &mut [RingBuf],
    target_channels: usize,
  ) {
    if target_channels == 0 || accum.is_empty() {
      return;
    }
    let frames = decoded.frames();

    macro_rules! process {
      ($buf:expr, $to_f32:expr) => {{
        let l = $buf.plane(0).unwrap_or(&[]);
        let r = if $buf.num_planes() > 1 {
          $buf.plane(1).unwrap_or(l)
        } else {
          l
        };
        if target_channels == 2 && accum.len() >= 2 {
          let frames_to_process = frames.min(l.len()).min(r.len());
          let l_slice = &l[..frames_to_process];
          let r_slice = &r[..frames_to_process];
          for (&ls, &rs) in l_slice.iter().zip(r_slice.iter()) {
            accum[0].push($to_f32(ls));
            accum[1].push($to_f32(rs));
          }
        } else {
          for i in 0..frames {
            let lv = l.get(i).map(|&s| $to_f32(s)).unwrap_or(0.0);
            let rv = r.get(i).map(|&s| $to_f32(s)).unwrap_or(0.0);
            if target_channels >= 1 {
              accum[0].push(lv);
            }
            if target_channels >= 2 && accum.len() >= 2 {
              accum[1].push(rv);
            }
          }
        }
      }};
    }

    match decoded {
      GenericAudioBufferRef::F32(b) => process!(b, |s: f32| s),
      GenericAudioBufferRef::F64(b) => process!(b, |s: f64| s as f32),
      GenericAudioBufferRef::S16(b) => process!(b, |s: i16| s as f32 / 32_767.0),
      GenericAudioBufferRef::U8(b) => process!(b, |s: u8| (s as f32 - 128.0) / 128.0),
      GenericAudioBufferRef::S24(b) => {
        process!(b, |s: i24| s.inner() as f32 / 8_388_607.0)
      }
      GenericAudioBufferRef::S32(b) => process!(b, |s: i32| s as f32 / 2_147_483_647.0),
      _ => {}
    }
  }
}

struct RingBuf {
  buf: Vec<f32>,
  read: usize,
  write: usize,
}

impl RingBuf {
  fn with_capacity(cap: usize) -> Self {
    Self {
      buf: Vec::with_capacity(cap),
      read: 0,
      write: 0,
    }
  }

  #[inline]
  fn len(&self) -> usize {
    self.write - self.read
  }

  #[inline]
  fn push(&mut self, val: f32) {
    if self.write >= self.buf.len() {
      self.buf.push(val);
    } else {
      self.buf[self.write] = val;
    }
    self.write += 1;
  }

  fn drain_to(&mut self, dst: &mut [f32]) -> usize {
    let n = dst.len().min(self.len());
    if n == 0 {
      return 0;
    }
    dst[..n].copy_from_slice(&self.buf[self.read..self.read + n]);
    self.read += n;
    if self.read > self.buf.len() / 2 {
      let remaining = self.len();
      self.buf.copy_within(self.read..self.write, 0);
      self.read = 0;
      self.write = remaining;
    }
    n
  }

  fn clear(&mut self) {
    self.read = 0;
    self.write = 0;
  }
}

fn process_resampler_chunk(
  resampler: &mut Fft<f32>,
  input: &[Vec<f32>],
  out_buf: &mut [Vec<f32>],
  output: &mut Vec<u8>,
  channels: usize,
  chunk_size: usize,
  partial_len: Option<usize>,
) -> napi::Result<usize> {
  let out_capacity = out_buf[0].len();

  let in_adapter = SequentialSliceOfVecs::new(input, channels, chunk_size)
    .map_err(|e| Error::from_reason(format!("Resample input adapter: {e}")))?;
  let mut out_adapter = SequentialSliceOfVecs::new_mut(out_buf, channels, out_capacity)
    .map_err(|e| Error::from_reason(format!("Resample output adapter: {e}")))?;

  let indexing = partial_len.map(|len| Indexing {
    input_offset: 0,
    output_offset: 0,
    partial_len: Some(len),
    active_channels_mask: None,
  });

  let (_, frames) = resampler
    .process_into_buffer(&in_adapter, &mut out_adapter, indexing.as_ref())
    .map_err(|e| Error::from_reason(format!("Resampler: {e}")))?;

  for f in 0..frames {
    for ch_buf in out_buf.iter().take(channels) {
      let sample = helpers::f32_to_i16(ch_buf.get(f).copied().unwrap_or(0.0));
      output.extend_from_slice(&sample.to_le_bytes());
    }
  }

  Ok(frames)
}

fn flush_resampler_tail(
  resampler: &mut Fft<f32>,
  accum: &mut [RingBuf],
  resampler_in: &mut [Vec<f32>],
  out_buf: &mut [Vec<f32>],
  output: &mut Vec<u8>,
  channels: usize,
  chunk_size: usize,
) -> napi::Result<()> {
  while accum[0].len() >= chunk_size {
    for ch in 0..channels {
      resampler_in[ch].resize(chunk_size, 0.0);
      accum[ch].drain_to(&mut resampler_in[ch]);
    }
    process_resampler_chunk(
      resampler,
      resampler_in,
      out_buf,
      output,
      channels,
      chunk_size,
      None,
    )?;
  }

  let remaining = accum[0].len();
  if remaining > 0 {
    for ch in 0..channels {
      resampler_in[ch].clear();
      resampler_in[ch].resize(chunk_size, 0.0);
      accum[ch].drain_to(&mut resampler_in[ch][..remaining]);
    }
    process_resampler_chunk(
      resampler,
      resampler_in,
      out_buf,
      output,
      channels,
      chunk_size,
      Some(remaining),
    )?;
  }

  Ok(())
}

#[napi(object)]
pub struct DecodeResult {
  pub samples: Buffer,
  pub sample_rate: u32,
  pub channels: u32,
}

/// Codec registry with every default Symphonia decoder plus Opus via libopus.
///
/// Symphonia has no first-party Opus support, so Ogg Opus streams fail with
/// "unsupported audio codec" unless the adapter is registered here.
static CODEC_REGISTRY: OnceLock<CodecRegistry> = OnceLock::new();

fn get_codec_registry() -> &'static CodecRegistry {
  CODEC_REGISTRY.get_or_init(|| {
    let mut registry = CodecRegistry::new();
    symphonia::default::register_enabled_codecs(&mut registry);
    registry.register_audio_decoder::<OpusDecoder>();
    registry
  })
}

/// Streaming Symphonia decoder.
///
/// Lifecycle/// 1. `push(chunk)` feeds compressed data.
/// 2. `initialize(hint?)` probes once enough header/metadata is available.
/// 3. `decode()` returns PCM when a full packet is available, or `null` when
///    more compressed input is needed.
/// 4. `closeInput()` signals end-of-input so the final packet/tail can flush.
/// 5. `free()` releases resources.
#[napi]
pub struct SymphoniaDecoder {
  buffer: SharedBuffer,

  format_reader: Option<Box<dyn FormatReader>>,
  audio_decoder: Option<Box<dyn AudioDecoder>>,
  track_id: Option<u32>,
  is_probed: bool,
  exhausted: bool,

  target_rate: u32,
  target_channels: usize,

  resampler: Option<Fft<f32>>,
  input_accumulator: Vec<RingBuf>,
  resampler_in: Vec<Vec<f32>>,
  resampler_out: Vec<Vec<f32>>,
  resample_chunk_size: usize,
  stream_decode_threshold: usize,

  final_output_buffer: Vec<u8>,
}

impl Default for SymphoniaDecoder {
  fn default() -> Self {
    Self::new()
  }
}

#[napi]
impl SymphoniaDecoder {
  #[napi(constructor)]
  pub fn new() -> Self {
    let target_channels = 2usize;
    Self {
      buffer: SharedBuffer::new(),
      format_reader: None,
      audio_decoder: None,
      track_id: None,
      is_probed: false,
      exhausted: false,
      target_rate: 48_000,
      target_channels,
      resampler: None,
      input_accumulator: (0..target_channels)
        .map(|_| RingBuf::with_capacity(RESAMPLE_CHUNK_SIZE * 2))
        .collect(),
      resampler_in: (0..target_channels)
        .map(|_| Vec::with_capacity(RESAMPLE_CHUNK_SIZE))
        .collect(),
      resampler_out: Vec::new(),
      resample_chunk_size: RESAMPLE_CHUNK_SIZE,
      stream_decode_threshold: STREAM_DECODE_THRESHOLD,
      final_output_buffer: Vec::with_capacity(MAX_OUTPUT_FRAMES * target_channels * 2),
    }
  }

  #[napi]
  pub fn push(&mut self, chunk: Buffer) -> Result<()> {
    self.buffer.push(chunk.as_ref());
    Ok(())
  }

  #[napi]
  pub fn close_input(&mut self) -> Result<()> {
    self.buffer.close();
    Ok(())
  }

  /// Returns the number of compressed bytes currently available to the reader.
  #[napi(getter)]
  pub fn buffered_bytes(&self) -> u32 {
    self.buffer.available_bytes() as u32
  }

  #[napi(getter)]
  pub fn is_probed(&self) -> bool {
    self.is_probed
  }

  /// Probe the format and set up the decoder when enough input is available.
  /// `codec_registry_hint` is an optional file extension (e.g. `"flac"`, `"mp3"`)
  /// that speeds up probing; pass `null` to auto-detect.
  ///
  /// `stream_decode_threshold` optionally overrides the number of buffered
  /// compressed bytes required before `decode()` starts producing PCM. Lower
  /// values reduce start-up latency on live streams (default: 16 KiB ≈ 1s at
  /// 128 kbps); pass `0` to decode as soon as a packet is available.
  #[napi]
  pub fn initialize(
    &mut self,
    codec_registry_hint: Option<String>,
    stream_decode_threshold: Option<u32>,
  ) -> Result<bool> {
    if self.is_probed {
      return Ok(true);
    }
    if let Some(threshold) = stream_decode_threshold {
      self.stream_decode_threshold = threshold as usize;
    }
    if !self.buffer.is_closed() && self.buffer.available_bytes() < PROBE_THRESHOLD {
      return Ok(false);
    }

    self.buffer.start_probe();

    let mss = MediaSourceStream::new(
      Box::new(StreamingSource {
        buffer: self.buffer.clone(),
      }),
      Default::default(),
    );

    let mut hint = Hint::new();
    if let Some(ext) = codec_registry_hint
      && !ext.is_empty()
    {
      hint.with_extension(&ext);
    }

    let reader = match symphonia::default::get_probe().probe(
      &hint,
      mss,
      FormatOptions::default(),
      MetadataOptions::default(),
    ) {
      Ok(reader) => reader,
      Err(SymphoniaError::IoError(e))
        if !self.buffer.is_closed()
          && (e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::UnexpectedEof) =>
      {
        self.buffer.reset_probe();
        return Ok(false);
      }
      Err(e) => {
        self.buffer.reset_probe();
        return Err(Error::from_reason(format!("Probe failed: {e}")));
      }
    };

    let track_info = reader
      .tracks()
      .iter()
      .find(|t| {
        t.codec_params
          .as_ref()
          .is_some_and(|cp| matches!(cp, CodecParameters::Audio(_)))
      })
      .map(|t| (t.id, t.codec_params.clone()));

    let Some((id, Some(CodecParameters::Audio(audio_params)))) = track_info else {
      return Err(Error::from_reason("No audio track found"));
    };

    self.configure_audio_decoder(id, &audio_params)?;
    self.format_reader = Some(reader);
    self.buffer.commit_probe();
    self.is_probed = true;
    Ok(true)
  }

  #[napi]
  pub fn decode(&mut self) -> Result<Option<DecodeResult>> {
    if !self.is_probed || self.exhausted {
      return Ok(None);
    }

    let mut reader = self
      .format_reader
      .take()
      .ok_or_else(|| Error::from_reason("Format reader gone"))?;
    let complete_reads = self.buffer.is_closed();
    let result = self.decode_from_reader(reader.as_mut(), complete_reads);
    self.format_reader = Some(reader);
    result
  }
}

impl SymphoniaDecoder {
  fn configure_audio_decoder(
    &mut self,
    id: u32,
    audio_params: &symphonia::core::codecs::audio::AudioCodecParameters,
  ) -> Result<()> {
    let decoder = get_codec_registry()
      .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
      .map_err(|e| Error::from_reason(format!("Failed to create decoder: {e}")))?;

    let source_rate = audio_params.sample_rate.unwrap_or(self.target_rate);
    if source_rate != self.target_rate {
      // new_custom with sub_chunks = 1 reproduces the v2 behavior (one FFT block
      // per chunk, BlackmanHarris2 window). v5's default `new()` would pick
      // chunk_size / 256 sub-chunks for lower delay; not what we had before.
      let resampler = Fft::<f32>::new_custom(
        source_rate as usize,      // input sample rate
        self.target_rate as usize, // output sample rate
        self.resample_chunk_size,  // input chunk size (FixedSync::Input)
        1,                         // sub_chunks (1 = one FFT block per chunk)
        self.target_channels,      // output channels
        WindowFunction::BlackmanHarris2,
        FixedSync::Input,
      )
      .map_err(|e| Error::from_reason(format!("Failed to create resampler: {e}")))?;

      // output_frames_next() is a prediction for the *next single call* and can
      // vary. output_frames_max() is the hard upper bound for any call. Passing
      // an undersized adapter to process_into_buffer() causes a panic because
      // rubato writes past the adapter's declared capacity.
      let out_max = resampler.output_frames_max();
      self.resampler_out = (0..self.target_channels)
        .map(|_| vec![0.0f32; out_max])
        .collect();

      self.resampler = Some(resampler);
    }

    self.track_id = Some(id);
    self.audio_decoder = Some(decoder);
    Ok(())
  }

  fn decode_from_reader(
    &mut self,
    reader: &mut dyn FormatReader,
    complete_reads: bool,
  ) -> Result<Option<DecodeResult>> {
    if self.exhausted {
      return Ok(None);
    }
    let audio_dec = self
      .audio_decoder
      .as_mut()
      .ok_or_else(|| Error::from_reason("Audio decoder gone"))?;
    let track_id = self
      .track_id
      .ok_or_else(|| Error::from_reason("Track ID not set"))?;

    let needs_resample = self.resampler.is_some();
    let chunk_size = self.resample_chunk_size;
    let target_channels = self.target_channels;
    let max_samples = MAX_OUTPUT_FRAMES * target_channels * 2; // bytes

    self.final_output_buffer.clear();

    loop {
      if needs_resample && let Some(ref mut rs) = self.resampler {
        while self.input_accumulator[0].len() >= chunk_size {
          for ch in 0..target_channels {
            self.resampler_in[ch].resize(chunk_size, 0.0);
            let n = self.input_accumulator[ch].drain_to(&mut self.resampler_in[ch]);
            debug_assert_eq!(n, chunk_size);
          }
          process_resampler_chunk(
            rs,
            &self.resampler_in,
            &mut self.resampler_out,
            &mut self.final_output_buffer,
            target_channels,
            chunk_size,
            None,
          )?;
          if self.final_output_buffer.len() >= max_samples {
            break;
          }
        }
        if self.final_output_buffer.len() >= max_samples {
          break;
        }
      }

      if !complete_reads && self.buffer.available_bytes() < self.stream_decode_threshold {
        break;
      }

      match reader.next_packet() {
        Ok(Some(packet)) => {
          if packet.track_id != track_id {
            continue;
          }
          match audio_dec.decode(&packet) {
            Ok(decoded) => {
              if !needs_resample {
                helpers::append_interleaved_from_decoded(
                  &decoded,
                  &mut self.final_output_buffer,
                  target_channels,
                );
                break; // one packet per call on the direct path
              } else {
                helpers::push_decoded_as_planar(
                  &decoded,
                  &mut self.input_accumulator,
                  target_channels,
                );
              }
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(Error::from_reason(format!("Decode error: {e}"))),
          }
        }

        Ok(None) => {
          self.exhausted = true;
          if let Some(ref mut rs) = self.resampler {
            flush_resampler_tail(
              rs,
              &mut self.input_accumulator,
              &mut self.resampler_in,
              &mut self.resampler_out,
              &mut self.final_output_buffer,
              target_channels,
              chunk_size,
            )?;
          }
          break;
        }

        // Treat unexpected EOF / WouldBlock gracefully rather than
        // propagating — flush whatever we have and stop.
        Err(SymphoniaError::IoError(e))
          if e.kind() == ErrorKind::UnexpectedEof || e.kind() == ErrorKind::WouldBlock =>
        {
          if complete_reads && !self.buffer.is_closed() {
            return Err(Error::from_reason(format!(
              "MP4 packet extraction failed: {e}"
            )));
          }
          if !complete_reads {
            break;
          }

          self.exhausted = true;
          if let Some(ref mut rs) = self.resampler {
            flush_resampler_tail(
              rs,
              &mut self.input_accumulator,
              &mut self.resampler_in,
              &mut self.resampler_out,
              &mut self.final_output_buffer,
              target_channels,
              chunk_size,
            )?;
          }
          break;
        }

        Err(e) => return Err(Error::from_reason(format!("Format reader error: {e}"))),
      }
    }

    if self.final_output_buffer.is_empty() {
      return Ok(None);
    }

    // Transferring the Vec into the Buffer is zero-copy on Node (external
    // buffer with a finalizer); Buffer::from(&[u8]) would memcpy instead.
    let out = std::mem::take(&mut self.final_output_buffer);

    Ok(Some(DecodeResult {
      samples: Buffer::from(out),
      sample_rate: self.target_rate,
      channels: target_channels as u32,
    }))
  }
}

#[napi]
impl SymphoniaDecoder {
  #[napi]
  pub fn flush(&mut self) -> Result<()> {
    self.buffer.clear();
    self.exhausted = false;
    self.format_reader = None;
    self.audio_decoder = None;
    self.track_id = None;
    self.is_probed = false;
    self.resampler = None;
    for ch in &mut self.input_accumulator {
      ch.clear();
    }
    for ch in &mut self.resampler_in {
      ch.clear();
    }
    self.resampler_out.clear();
    self.final_output_buffer.clear();
    Ok(())
  }

  #[napi]
  pub fn free(&mut self) {
    self.buffer.clear();
    self.exhausted = false;
    self.format_reader = None;
    self.audio_decoder = None;
    self.resampler = None;
    self.track_id = None;
    self.is_probed = false;
    self.input_accumulator = Vec::new();
    self.resampler_in = Vec::new();
    self.resampler_out = Vec::new();
    self.final_output_buffer = Vec::new();
  }
}
