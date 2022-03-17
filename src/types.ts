import * as tf from '@tensorflow/tfjs'

export interface LayerFilters {
  [k: string]: Array<tf.Tensor[]>
}
export interface Button {
  selector: string
  eventListener: string
  callback: () => void
}

export interface PixelData {
  p: ImageData
  x: number
  y: number
}

export type DrawCallback = (k: string, data: PixelData) => void

export interface DecodedKernel {
  layer: number
  filter: number
  kernel: number
}
