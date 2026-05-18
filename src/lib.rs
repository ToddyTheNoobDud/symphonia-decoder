#![deny(clippy::all)]
#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]

use std::collections::VecDeque;
use std::io::ErrorKind;
use std::io::{Error as IoError, Read, Seek, SeekFrom};
use std::slice;
use std::sync::{Arc, Mutex};

use napi::bindgen_prelude::*;
use napi_derive::napi;

use rubato::{
  Fft, FixedSync, Indexing, Resampler, audioadapter_buffers::direct::SequentialSliceOfVecs,
};

use symphonia::core::audio::sample::i24;
use symphonia::core::audio::{Audio, GenericAudioBufferRef};
use symphonia::core::codecs::CodecParameters;
use symphonia::core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;

const SAMPLE_MAX: f32 = 32767.0;
const INITIAL_BUFFER_CAPACITY: usize = 1024 * 1024;
const PROBE_THRESHOLD: usize = 1024;
const STREAM_DECODE_THRESHOLD: usize = 1024 * 1024;
const RESAMPLE_CHUNK_SIZE: usize = 4096;
const MAX_OUTPUT_FRAMES: usize = 4096;

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
    out: &mut Vec<i16>,
    target_channels: usize,
  ) {
    if target_channels == 0 {
      return;
    }
    let frames = decoded.frames();
    out.reserve(frames * target_channels);

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
            out.push(f32_to_i16($to_f32(ls)));
            out.push(f32_to_i16($to_f32(rs)));
          }
        } else {
          for i in 0..frames {
            let lv = l.get(i).map(|&s| $to_f32(s)).unwrap_or(0.0);
            let rv = r.get(i).map(|&s| $to_f32(s)).unwrap_or(0.0);
            if target_channels >= 1 {
              out.push(f32_to_i16(lv));
            }
            if target_channels >= 2 {
              out.push(f32_to_i16(rv));
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
  output: &mut Vec<i16>,
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
      output.push(helpers::f32_to_i16(ch_buf.get(f).copied().unwrap_or(0.0)));
    }
  }

  Ok(frames)
}

fn flush_resampler_tail(
  resampler: &mut Fft<f32>,
  accum: &mut [RingBuf],
  resampler_in: &mut [Vec<f32>],
  out_buf: &mut [Vec<f32>],
  output: &mut Vec<i16>,
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

/// Streaming Symphonia decoder.
///
/// Lifecycle
/// 1. `push(chunk)` feeds compressed data.
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

  final_output_buffer: Vec<i16>,
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
      final_output_buffer: Vec::with_capacity(MAX_OUTPUT_FRAMES * target_channels),
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
  #[napi]
  pub fn initialize(&mut self, codec_registry_hint: Option<String>) -> Result<bool> {
    if self.is_probed {
      return Ok(true);
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

    let decoder = symphonia::default::get_codecs()
      .make_audio_decoder(&audio_params, &AudioDecoderOptions::default())
      .map_err(|e| Error::from_reason(format!("Failed to create decoder: {e}")))?;

    let source_rate = audio_params.sample_rate.unwrap_or(self.target_rate);
    if source_rate != self.target_rate {
      let resampler = Fft::<f32>::new(
        source_rate as usize,      // input sample rate
        self.target_rate as usize, // output sample rate
        self.resample_chunk_size,  // input chunk size (FixedSync::Input)
        1,                         // sub_chunks quality factor
        self.target_channels,      // output channels
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

    let reader = self
      .format_reader
      .as_mut()
      .ok_or_else(|| Error::from_reason("Format reader gone"))?;
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
    let max_samples = MAX_OUTPUT_FRAMES * target_channels;

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

      if !self.buffer.is_closed() && self.buffer.available_bytes() < STREAM_DECODE_THRESHOLD {
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
          if !self.buffer.is_closed() {
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

    let byte_len = self.final_output_buffer.len() * 2;
    let ptr = self.final_output_buffer.as_ptr() as *const u8;
    let bytes = unsafe { slice::from_raw_parts(ptr, byte_len) };

    Ok(Some(DecodeResult {
      samples: Buffer::from(bytes),
      sample_rate: self.target_rate,
      channels: target_channels as u32,
    }))
  }

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
