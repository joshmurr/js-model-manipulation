import * as tf from '@tensorflow/tfjs'

export interface LayerFilters {
  [k: string]: Array<tf.Tensor[]>
}
export interface Button {
  selector: string
  eventListener: string
  callback: () => void
}
