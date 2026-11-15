import test from 'ava'

import { demuxMp4Aac, openMp4Audio, PcmResampler, SymphoniaDecoder } from '../index.js'

test('SymphoniaDecoder can be instantiated', (t) => {
  const decoder = new SymphoniaDecoder()
  t.truthy(decoder)
  t.is(decoder.isProbed, false)
  t.is(decoder.bufferedBytes, 0)
})

test('push and closeInput work', (t) => {
  const decoder = new SymphoniaDecoder()
  decoder.push(Buffer.from([0, 1, 2, 3]))
  t.is(decoder.bufferedBytes, 4)
  decoder.closeInput()
  t.pass()
})

test('MP4 exports reject malformed input', async (t) => {
  const input = Buffer.alloc(32)
  t.throws(() => demuxMp4Aac(input), { message: /Invalid MP4/ })
  await t.throwsAsync(
    () => openMp4Audio(input.length, async (_error, [offset, length]) => input.subarray(offset, offset + length)),
    { message: /Invalid MP4/ },
  )
})

test('PCM resampler drains the expected number of stereo frames and closes', (t) => {
  const resampler = new PcmResampler(24000, 'fastest')
  const pcm = resampler.process(new Float32Array(300).fill(0.5), 1)
  const tail = resampler.finish()

  t.is(pcm.length + tail.length, 600 * 2 * 2)
  t.throws(() => resampler.process(new Float32Array(1), 1), { message: /resampler is closed/ })
})
