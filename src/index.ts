/* eslint-disable @typescript-eslint/no-unused-vars */

import * as tf from '@tensorflow/tfjs'
import CNN from './CNN'
import Gen from './Generator'
import GUI from './GUI'
import Editor from './Editor'
import { MnistData } from './data.js'
import DataLoader from './DataLoader'
import fashion_mnist from './data/fashion_mnist.png'
import fashion_mnist_labels from './data/fashion_mnist_labels.npy'
import mnist from './data/mnist.png'
import mnist_labels from './data/mnist_labels_uint8.dat'

import { Button, Checkbox, DataLoaderOpts } from './types'

import './styles.scss'

async function run() {
  const data = new MnistData()
  await data.load()

  const gui = new GUI()

  const buttons: Button[] = [
    {
      selector: '.play-btn',
      eventListener: 'mouseup',
      callback: trainModel,
    },
    {
      selector: '.update-btn',
      eventListener: 'mouseup',
      callback: updateGUI,
    },
    {
      selector: '.predict-btn',
      eventListener: 'mouseup',
      callback: predict,
    },
    {
      selector: '.show-examples',
      eventListener: 'mouseup',
      callback: showExamples,
    },
  ]

  const checkboxes: Checkbox[] = [
    {
      name: 'diff',
      selector: 'input[name="diff"]',
    },
  ]

  gui.initButtons(buttons)
  gui.initCheckboxes(checkboxes)

  const dataLoader = new DataLoader({
    imagesPath: fashion_mnist,
    labelsPath: fashion_mnist_labels,
    ratio: 6 / 7,
    numClasses: 10,
  })
  await dataLoader.loadImages()

  //const model = await new CNN(gui)
  //await model.warm()

  const model = await new Gen(gui)
  await model.warm()

  const editor = new Editor()
  await gui.initModel(model, editor)
  gui.initOutput(model)
  gui.initChart('loss')

  async function showExamples() {
    await gui.showExamples(dataLoader)
  }

  async function updateGUI() {
    gui.update(model)
  }

  async function trainModel() {
    const playBtn = <HTMLElement>document.querySelector('.play-btn')
    if (model.isTraining) {
      model.isTraining = false
      playBtn.innerText = 'Play'
    } else {
      model.isTraining = true
      playBtn.innerText = 'Pause'
      await model.train()
    }
  }

  async function predict() {
    const pred = await model.doPrediction()
    tf.browser.toPixels(pred as tf.Tensor3D, gui.output.modelOutput)
  }
}

document.addEventListener('DOMContentLoaded', run)
