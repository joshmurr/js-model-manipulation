/* eslint-disable @typescript-eslint/no-unused-vars */

import * as tf from '@tensorflow/tfjs'
import { MnistData } from './data.js'

type LayerFilters = { [k: string]: Array<tf.Tensor[]> }

async function showExamples(data: MnistData) {
  // Get the examples
  const examples = data.nextTestBatch(20)
  const numExamples = examples.xs.shape[0]

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1])
    })

    const canvas = document.createElement('canvas')
    canvas.width = 28
    canvas.height = 28
    canvas.style.margin = '4px'
    await tf.browser.toPixels(imageTensor as tf.Tensor2D, canvas)
    document.body.appendChild(canvas)

    imageTensor.dispose()
  }
}

function getModel() {
  const model = tf.sequential()

  const IMAGE_WIDTH = 28
  const IMAGE_HEIGHT = 28
  const IMAGE_CHANNELS = 1

  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
    })
  )

  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
    })
  )
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

  model.add(tf.layers.flatten())

  const NUM_OUTPUT_CLASSES = 10
  model.add(
    tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax',
    })
  )

  const optimizer = tf.train.adam()
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  })

  return model
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

async function getLayers(model: tf.LayersModel, data: MnistData) {
  const layers = []

  for (let i = 0; i <= model.layers.length - 1; i++) {
    const layer = model.getLayer(undefined, i)
    if (layer.name.includes('conv2d')) {
      layers.push(layer)
    }
  }

  return layers
}

async function getFilters(layers: Array<tf.layers.Layer>) {
  const layerFilters: LayerFilters = {}

  for (const layer of layers) {
    const filters = []

    const kernelDesc = await layer.getWeights()[0]
    const kernelData = await kernelDesc.data()

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

async function renderKernels(filterLayers: LayerFilters) {
  const renderDiv = document.getElementById('kernels')
  const row = document.createElement('div')

  for (const layerName in filterLayers) {
    const filterArray = filterLayers[layerName]
    const title = document.createElement('h4')
    title.innerText = layerName
    row.appendChild(title)
    for (const f of filterArray) {
      for (const k of f) {
        //const rk = tf.image.resizeNearestNeighbor(k.squeeze(-1), [28,28])
        const [w, h, d, _] = k.shape
        const c = document.createElement('canvas')
        c.width = w
        c.height = h
        const ctx = c.getContext('2d')

        const img = new ImageData(w, h)

        const data = k.dataSync()

        for (let x = 0; x < w; x++) {
          for (let y = 0; y < h; y++) {
            const ix = (y * w + x) * 4

            const iv = y * w + x
            img.data[ix + 0] = Math.floor(255 * data[iv])
            img.data[ix + 1] = Math.floor(255 * data[iv])
            img.data[ix + 2] = Math.floor(255 * data[iv])
            img.data[ix + 3] = 255
          }
        }

        ctx.putImageData(img, 0, 0)
        row.appendChild(c)
      }
      renderDiv.appendChild(row)
    }
  }
}

async function train(model: tf.LayersModel, data: MnistData) {
  const onEpochEnd = async (epoch: number, logs: tf.Logs) => {
    if (epoch % 2 === 0) {
      console.log('Rendering...')
      const layers: Array<tf.layers.Layer> = await getLayers(model, data)
      const filters = await getFilters(layers)
      await renderKernels(filters)
    }
  }

  const BATCH_SIZE = 512
  const TRAIN_DATA_SIZE = 5500
  const TEST_DATA_SIZE = 1000

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE)
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels]
  })

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE)
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels]
  })

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 1,
    shuffle: true,
    callbacks: { onEpochEnd },
  })
}

const classNames = [
  'Zero',
  'One',
  'Two',
  'Three',
  'Four',
  'Five',
  'Six',
  'Seven',
  'Eight',
  'Nine',
]

function doPrediction(
  model: tf.LayersModel,
  data: MnistData,
  testDataSize = 500
) {
  const IMAGE_WIDTH = 28
  const IMAGE_HEIGHT = 28
  const testData = data.nextTestBatch(testDataSize)
  const testxs = testData.xs.reshape([
    testDataSize,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    1,
  ])
  const labels = testData.labels.argMax(-1)
  const preds = model.predict(testxs) as tf.Tensor
  const pred = preds.argMax(-1)

  testxs.dispose()
  return [pred, labels]
}

async function run() {
  const data = new MnistData()
  await data.load()

  const renderDiv = document.createElement('div')
  renderDiv.id = 'kernels'
  document.body.appendChild(renderDiv)

  //await showExamples(data);
  const model = getModel()

  await train(model, data)
}

document.addEventListener('DOMContentLoaded', run)
