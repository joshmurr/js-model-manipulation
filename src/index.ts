/* eslint-disable @typescript-eslint/no-unused-vars */

import * as tf from '@tensorflow/tfjs'
import CNN from './CNN'
import { MnistData } from './data.js'

import './styles.scss'

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

async function renderKernels(filterLayers: LayerFilters) {
  const renderDiv = document.getElementById('kernels')

  for (const layerName in filterLayers) {
    const filterArray = filterLayers[layerName]
    const title = document.createElement('h4')
    title.innerText = layerName
    renderDiv.appendChild(title)
    for (const f of filterArray) {
      const row = document.createElement('div')
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
            img.data[ix + 0] = Math.floor(127 * (data[iv] + 1))
            img.data[ix + 1] = Math.floor(127 * (data[iv] + 1))
            img.data[ix + 2] = Math.floor(127 * (data[iv] + 1))
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

async function train(model: CNN, data: MnistData) {
  const onEpochEnd = async (epoch: number, logs: tf.Logs) => {
    if (epoch % 10 === 0) {
      console.log('Rendering...')
      const layers: Array<tf.layers.Layer> = await model.getLayers()
      const filters = await model.getFilters(layers)
      await renderKernels(filters)
    }
  }

  const BATCH_SIZE = 8 // 512
  const TRAIN_DATA_SIZE = 8 // 5500
  const TEST_DATA_SIZE = 8 // 1000

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE)
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels]
  })

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE)
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels]
  })

  return model.model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 1,
    shuffle: true,
    callbacks: { onEpochEnd },
  })
}

async function run() {
  const data = new MnistData()
  await data.load()

  const renderDiv = document.createElement('div')
  renderDiv.id = 'kernels'
  document.body.appendChild(renderDiv)

  //await showExamples(data);
  //const model = getModel()

  const model = new CNN()

  await train(model, data)
}

document.addEventListener('DOMContentLoaded', run)
