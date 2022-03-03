import * as tf from '@tensorflow/tfjs'

import { LayerFilters } from './types'

export default class Model {
  public model: tf.Sequential

  protected build() {
    this.model = tf.sequential()
  }

  async getLayers() {
    const layers = []

    for (let i = 0; i <= this.model.layers.length - 1; i++) {
      const layer = this.model.getLayer(undefined, i)
      if (layer.name.includes('conv2d')) {
        layers.push(layer)
      }
    }

    return layers
  }

  async getFilters(layers: Array<tf.layers.Layer>) {
    const layerFilters: LayerFilters = {}

    for (const layer of layers) {
      const filters = []

      const kernelDesc = await layer.getWeights()[0]

      const shape = kernelDesc.shape
      const numFilters = shape[3]

      for (let i = 0; i < numFilters; i++) {
        const kernels = []
        for (let j = 0; j < shape[2]; j++) {
          const newShape = [shape[0], shape[1], j + 1, i + 1]
          const chunk = kernelDesc.stridedSlice(
            [0, 0, j, i],
            newShape,
            [1, 1, 1, 1]
          )

          kernels.push(chunk)
        }

        filters.push(kernels)
      }
      const k: keyof LayerFilters = layer.name
      layerFilters[k] = filters
    }
    return layerFilters
  }

  //function getLayerOutputs(model: tf.LayersModel, data: MnistData) {
  //return tf.tidy(() => {
  //const [X, y] = tf.tidy(() => {
  //const d = data.single()
  //return [d.xs.reshape([1, 28, 28, 1]), d.labels]
  //})

  //const withIntermediate = true

  //if (withIntermediate === true) {
  //const outputNames = []

  //for (let i = 0; i <= model.layers.length - 1; i++) {
  //const layer = model.getLayer(undefined, i)
  //console.log(layer)
  //const name = layer.output.name
  //outputNames.push(name)
  //}
  //} else {
  //outputNames = [
  //model.getLayer(undefined, model.layers.length - 1).output.name,
  //]
  //}

  //const outputs = model.execute(X, outputNames)
  ////this.renderEverything(outputs);
  //return outputs
  //})
  //}
}