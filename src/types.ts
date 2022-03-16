import * as tf from '@tensorflow/tfjs'

export interface LayerFilters {
  [k: string]: Array<tf.Tensor[]>
}
export interface Button {
  selector: string
  eventListener: string
  callback: () => void
}

export type InputShape = [number, number, number, number]

export interface PixelData {
  p: ImageData
  x: number
  y: number
}

export type ModelCallback = (k: string, data: PixelData) => void
