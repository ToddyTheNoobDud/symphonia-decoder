use napi::bindgen_prelude::*;
use napi_derive::napi;
use rubato::{
  Async, FixedAsync, Indexing, PolynomialDegree, Resampler, SincInterpolationParameters,
  WindowFunction, audioadapter_buffers::direct::InterleavedSlice,
};

const BLOCK: usize = 256;
const OUTPUT_RATE: u32 = 48_000;

#[napi]
pub struct PcmResampler {
  rate: u32,
  converter: Option<Async<f32>>,
  pending: Vec<f32>,
  scratch: Vec<f32>,
  input_frames: u64,
  output_frames: u64,
  delay: usize,
}

/// Converts FAAD2's interleaved float PCM to 48 kHz stereo signed 16-bit PCM.
#[napi]
impl PcmResampler {
  #[napi(constructor)]
  pub fn new(rate: u32, quality: String) -> Result<Self> {
    if !(7350..=192000).contains(&rate) {
      return Err(Error::from_reason("Unsupported AAC PCM sample rate"));
    }
    let converter = if rate == OUTPUT_RATE {
      None
    } else {
      let ratio = f64::from(OUTPUT_RATE) / f64::from(rate);
      let resampler = match quality.as_str() {
        "linear" | "zero order holder" => Async::new_poly(
          ratio,
          1.0,
          if quality == "linear" {
            PolynomialDegree::Linear
          } else {
            PolynomialDegree::Nearest
          },
          BLOCK,
          2,
          FixedAsync::Input,
        ),
        _ => {
          let length = match quality.as_str() {
            "best" => 256,
            "medium" => 128,
            _ => 64,
          };
          let parameters =
            SincInterpolationParameters::new(length, WindowFunction::BlackmanHarris2);
          Async::new_sinc(ratio, 1.0, &parameters, BLOCK, 2, FixedAsync::Input)
        }
      }
      .map_err(native_error)?;
      Some(resampler)
    };
    let delay = converter.as_ref().map_or(0, Resampler::output_delay);
    let scratch = converter
      .as_ref()
      .map_or_else(Vec::new, |r| vec![0.0; r.output_frames_max() * 2]);
    Ok(Self {
      rate,
      converter,
      pending: Vec::with_capacity(BLOCK * 2),
      scratch,
      input_frames: 0,
      output_frames: 0,
      delay,
    })
  }

  #[napi]
  pub fn process(&mut self, pcm: Float32Array, channels: u32) -> Result<Buffer> {
    if self.rate == 0 {
      return Err(Error::from_reason("AAC PCM resampler is closed"));
    }
    let channels = channels as usize;
    if channels == 0 || !pcm.len().is_multiple_of(channels) {
      return Err(Error::from_reason("Invalid interleaved AAC PCM"));
    }
    let frames = pcm.len() / channels;
    self.input_frames += frames as u64;
    let mut output = Vec::new();
    for frame in pcm.chunks_exact(channels) {
      let [left, right] = stereo(frame);
      if self.converter.is_none() {
        append_pcm(&mut output, &[left, right]);
      } else {
        self.pending.extend([left, right]);
        if self.pending.len() == BLOCK * 2 {
          self.convert(None, &mut output)?;
        }
      }
    }
    Ok(output.into())
  }

  #[napi]
  pub fn finish(&mut self) -> Result<Buffer> {
    let mut output = Vec::new();
    if self.converter.is_some() {
      while self.output_frames < self.expected_frames() {
        self.convert(Some(self.pending.len() / 2), &mut output)?;
      }
    }
    self.close();
    Ok(output.into())
  }

  #[napi]
  pub fn close(&mut self) {
    self.rate = 0;
    self.converter = None;
    self.pending = Vec::new();
    self.scratch = Vec::new();
    self.input_frames = 0;
    self.output_frames = 0;
    self.delay = 0;
  }
}

impl PcmResampler {
  fn expected_frames(&self) -> u64 {
    (self.input_frames * u64::from(OUTPUT_RATE) + u64::from(self.rate) / 2) / u64::from(self.rate)
  }

  fn convert(&mut self, partial_len: Option<usize>, output: &mut Vec<u8>) -> Result<()> {
    let source =
      InterleavedSlice::new(&self.pending, 2, self.pending.len() / 2).map_err(native_error)?;
    let frames = self.scratch.len() / 2;
    let mut target =
      InterleavedSlice::new_mut(&mut self.scratch, 2, frames).map_err(native_error)?;
    let indexing = partial_len.map(|frames| Indexing {
      input_offset: 0,
      output_offset: 0,
      partial_len: Some(frames),
      active_channels_mask: None,
    });
    let converter = self.converter.as_mut().expect("configured resampler");
    let (_, written) = converter
      .process_into_buffer(&source, &mut target, indexing.as_ref())
      .map_err(native_error)?;
    let skip = written.min(self.delay);
    self.delay -= skip;
    let frames =
      ((written - skip) as u64).min(self.expected_frames().saturating_sub(self.output_frames));
    self.output_frames += frames;
    append_pcm(
      output,
      &self.scratch[skip * 2..(skip + frames as usize) * 2],
    );
    self.pending.clear();
    Ok(())
  }
}

fn native_error(error: impl std::fmt::Display) -> Error {
  Error::from_reason(format!("AAC PCM resampling: {error}"))
}

fn append_pcm(output: &mut Vec<u8>, pcm: &[f32]) {
  output.reserve(pcm.len() * 2);
  for &sample in pcm {
    output.extend_from_slice(&((sample.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
  }
}

fn stereo(frame: &[f32]) -> [f32; 2] {
  use std::f32::consts::FRAC_1_SQRT_2;
  let mix = FRAC_1_SQRT_2;
  let (left, right) = match frame {
    [mono] => (*mono, *mono),
    [c, l, r] => (l + c * mix, r + c * mix),
    [c, l, r, center_surround] => {
      let shared = c * mix + center_surround * mix * 0.5;
      (l + shared, r + shared)
    }
    [c, l, r, ls, rs] => (l + (c + ls) * mix, r + (c + rs) * mix),
    [c, l, r, ls, rs, lfe] => {
      let shared = c * mix + lfe * 0.5;
      (l + ls * mix + shared, r + rs * mix + shared)
    }
    [left, right, ..] => (*left, *right),
    [] => unreachable!(),
  };
  [left.clamp(-1.0, 1.0), right.clamp(-1.0, 1.0)]
}
