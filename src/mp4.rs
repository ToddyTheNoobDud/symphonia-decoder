use std::io::{self, Read, Seek, SeekFrom};

#[napi_derive::napi(object)]
#[derive(Clone)]
pub struct Mp4AudioRange {
  pub start: f64,
  pub length: u32,
}

struct Atom {
  kind: [u8; 4],
  start: u64,
  payload: u64,
  end: u64,
}

fn atom_at(source: &mut (impl Read + Seek), start: u64, limit: u64) -> io::Result<Option<Atom>> {
  if limit.saturating_sub(start) < 8 {
    return Ok(None);
  }
  source.seek(SeekFrom::Start(start))?;
  let mut header = [0; 8];
  source.read_exact(&mut header)?;
  let (size, header_len) = match u32::from_be_bytes(header[..4].try_into().unwrap()) {
    0 => (limit - start, 8),
    1 if limit - start >= 16 => {
      let mut extended = [0; 8];
      source.read_exact(&mut extended)?;
      (u64::from_be_bytes(extended), 16)
    }
    size => (u64::from(size), 8),
  };
  if size < header_len || size > limit - start {
    return Ok(None);
  }
  Ok(Some(Atom {
    kind: header[4..].try_into().unwrap(),
    start,
    payload: start + header_len,
    end: start + size,
  }))
}

fn find_atom(
  source: &mut (impl Read + Seek),
  mut start: u64,
  limit: u64,
  kind: &[u8; 4],
) -> io::Result<Option<Atom>> {
  while let Some(atom) = atom_at(source, start, limit)? {
    if &atom.kind == kind {
      return Ok(Some(atom));
    }
    start = atom.end;
  }
  Ok(None)
}

pub(crate) fn track_layout(
  source: &mut (impl Read + Seek),
  length: u64,
) -> io::Result<(Vec<u64>, Vec<Mp4AudioRange>)> {
  let Some(movie) = find_atom(source, 0, length, b"moov")? else {
    return Ok((Vec::new(), Vec::new()));
  };

  let mut video_types = Vec::new();
  let mut audio_tracks = 0;
  let mut audio_media = None;
  let mut offset = movie.payload;
  while let Some(atom) = atom_at(source, offset, movie.end)? {
    offset = atom.end;
    if &atom.kind == b"mvex" {
      return Ok((Vec::new(), Vec::new()));
    }
    if &atom.kind != b"trak" {
      continue;
    }
    let Some(media) = find_atom(source, atom.payload, atom.end, b"mdia")? else {
      continue;
    };
    let Some(handler) = find_atom(source, media.payload, media.end, b"hdlr")? else {
      continue;
    };
    if handler.end - handler.payload < 12 {
      continue;
    }
    source.seek(SeekFrom::Start(handler.payload + 8))?;
    let mut kind = [0; 4];
    source.read_exact(&mut kind)?;
    match &kind {
      b"vide" => video_types.push(atom.start + 4),
      b"soun" => {
        audio_tracks += 1;
        audio_media = Some(media);
      }
      _ => {}
    }
  }

  /* INFO: Symphonia reads every track it sees. Mask only video track headers
  in ordinary MP4s, preserving every byte offset in the audio sample table.
  Fragmented MP4s keep their track references in moof and mvex. */
  if audio_tracks == 0 {
    video_types.clear();
  }
  let mut ranges = Vec::new();
  if audio_tracks == 1
    && !video_types.is_empty()
    && let Some(media) = audio_media
    && let Some(mdat) = find_atom(source, 0, movie.start, b"mdat")?
    && mdat.end == movie.start
  {
    ranges = single_sample_chunks(source, media, mdat)?;
  }
  Ok((video_types, ranges))
}

fn single_sample_chunks(
  source: &mut (impl Read + Seek),
  media: Atom,
  mdat: Atom,
) -> io::Result<Vec<Mp4AudioRange>> {
  let Some(minf) = find_atom(source, media.payload, media.end, b"minf")? else {
    return Ok(Vec::new());
  };
  let Some(table) = find_atom(source, minf.payload, minf.end, b"stbl")? else {
    return Ok(Vec::new());
  };
  let Some(mapping) = find_atom(source, table.payload, table.end, b"stsc")? else {
    return Ok(Vec::new());
  };
  if mapping.end - mapping.payload != 20 {
    return Ok(Vec::new());
  }
  source.seek(SeekFrom::Start(mapping.payload + 4))?;
  let mut values = [0; 12];
  source.read_exact(&mut values)?;
  if values != [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1] {
    return Ok(Vec::new());
  }

  let mut offsets = find_atom(source, table.payload, table.end, b"stco")?;
  if offsets.is_none() {
    offsets = find_atom(source, table.payload, table.end, b"co64")?;
  }
  let Some(offsets) = offsets else {
    return Ok(Vec::new());
  };
  let Some(sizes) = find_atom(source, table.payload, table.end, b"stsz")? else {
    return Ok(Vec::new());
  };
  let width = if &offsets.kind == b"co64" { 8 } else { 4 };
  if offsets.end - offsets.payload < 8 || sizes.end - sizes.payload < 12 {
    return Ok(Vec::new());
  }
  source.seek(SeekFrom::Start(offsets.payload + 4))?;
  let mut word = [0; 4];
  source.read_exact(&mut word)?;
  let count = u32::from_be_bytes(word) as usize;
  if !(2..=32_768).contains(&count) || offsets.end - offsets.payload != (8 + count * width) as u64 {
    return Ok(Vec::new());
  }
  let mut positions = vec![0; count * width];
  source.read_exact(&mut positions)?;
  source.seek(SeekFrom::Start(sizes.payload + 4))?;
  source.read_exact(&mut word)?;
  let default_size = u32::from_be_bytes(word);
  source.read_exact(&mut word)?;
  let size_bytes = if default_size == 0 { count * 4 } else { 0 };
  if u32::from_be_bytes(word) as usize != count
    || sizes.end - sizes.payload != (12 + size_bytes) as u64
  {
    return Ok(Vec::new());
  }
  let mut lengths = vec![0; size_bytes];
  source.read_exact(&mut lengths)?;

  let mut ranges = Vec::with_capacity(count);
  let mut previous_end = mdat.payload;
  for (index, position) in positions.chunks_exact(width).enumerate() {
    let start = if width == 8 {
      u64::from_be_bytes(position.try_into().unwrap())
    } else {
      u64::from(u32::from_be_bytes(position.try_into().unwrap()))
    };
    let length = if default_size == 0 {
      u32::from_be_bytes(lengths[index * 4..index * 4 + 4].try_into().unwrap())
    } else {
      default_size
    };
    if start < previous_end
      || !(1..=4096).contains(&length)
      || mdat.end.saturating_sub(start) < u64::from(length)
    {
      return Ok(Vec::new());
    }
    previous_end = start + u64::from(length);
    ranges.push(Mp4AudioRange {
      start: start as f64,
      length,
    });
  }
  Ok(ranges)
}
