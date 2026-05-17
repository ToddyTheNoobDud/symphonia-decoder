import test from 'ava'

import { SymphoniaDecoder } from '../index'

test('SymphoniaDecoder can be instantiated', (t) => {
  const decoder = new SymphoniaDecoder()
  t.truthy(decoder)
  t.is(decoder.isProbed, false)
  t.is(decoder.bufferedBytes, 0)
})

test('push and closeInput work', (t) => {
  const decoder = new SymphoniaDecoder()
  decoder.push(new Uint8Array([0, 1, 2, 3]))
  t.is(decoder.bufferedBytes, 4)
  decoder.closeInput()
  t.pass()
})
