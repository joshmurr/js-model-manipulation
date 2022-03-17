import * as tf from '@tensorflow/tfjs'

import { LayerFilters, DecodedKernel } from './types'

export default class Model {
  public net: tf.LayersModel | tf.Sequential
  protected training = false
  protected layers: Array<tf.layers.Layer>
  protected layerFilters: LayerFilters
  protected layerNames: string[]

  protected build() {
    this.net = tf.sequential()
  }

  protected async storeLayers() {
    this.layers = await this.getLayers()
    this.layerFilters = await this.getFilters(this.layers)
    this.layerNames = Object.keys(this.layerFilters)
  }

  public get layerInfo() {
    return {
      layers: this.layers,
      layerFilters: this.layerFilters,
      layerNames: this.layerNames,
    }
  }

  async getLayers() {
    const layers = []

    for (let i = 0; i <= this.net.layers.length - 1; i++) {
      const layer = this.net.getLayer(undefined, i)
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

  public set isTraining(training: boolean) {
    this.training = training
  }

  public get isTraining() {
    return this.training
  }

  public async setKernel(kernelInfo: DecodedKernel, data: ImageData) {
    const layers: Array<tf.layers.Layer> = await this.getLayers()
    const layerFilters: LayerFilters = await this.getFilters(layers)
    const layerNames = Object.keys(layerFilters)

    const layer = layerFilters[layerNames[kernelInfo.layer]]
    const filter = layer[kernelInfo.filter]
    filter[kernelInfo.kernel] = this.imageToTensor(data)

    layerNames.forEach((layerName) => {
      const filterArray = layerFilters[layerName]
      const filterStack: tf.Tensor[] = []
      filterArray.forEach((filter) => {
        filterStack.push(tf.stack(filter, -1).squeeze([2, 3]))
      })
      const layerStack = tf.stack(filterStack, -1) //.squeeze([-1])

      const layer = this.net.getLayer(layerName)
      layer.setWeights([layerStack]) // TODO: Add biases as well
    })
  }

  private imageToTensor(data: ImageData): tf.Tensor {
    const grayscale = data.data.reduce((acc, p, i) => {
      if (i % 4 === 0) {
        acc.push(i)
      }
      return acc
    }, [])

    const tensor = tf.tensor(grayscale).reshape([data.width, data.height, 1, 1])

    return tensor
  }

  //function getLayerOutputs(net: tf.LayersModel, data: MnistData) {
  //return tf.tidy(() => {
  //const [X, y] = tf.tidy(() => {
  //const d = data.single()
  //return [d.xs.reshape([1, 28, 28, 1]), d.labels]
  //})

  //const withIntermediate = true

  //if (withIntermediate === true) {
  //const outputNames = []

  //for (let i = 0; i <= net.layers.length - 1; i++) {
  //const layer = net.getLayer(undefined, i)
  //console.log(layer)
  //const name = layer.output.name
  //outputNames.push(name)
  //}
  //} else {
  //outputNames = [
  //net.getLayer(undefined, net.layers.length - 1).output.name,
  //]
  //}

  //const outputs = net.execute(X, outputNames)
  ////this.renderEverything(outputs);
  //return outputs
  //})
  //}
}
