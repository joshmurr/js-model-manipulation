import * as tf from '@tensorflow/tfjs'

export type LayerFilters = { [k: string]: Array<tf.Tensor[]> }
