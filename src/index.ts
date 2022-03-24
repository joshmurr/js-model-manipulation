/* eslint-disable @typescript-eslint/no-unused-vars */

import * as tf from '@tensorflow/tfjs'
import CNN from './CNN'
import Gen from './Generator'
import GUI from './GUI'
import Editor from './Editor'
import { MnistData } from './data.js'
import DataLoader from './DataLoader'
//import fashion_mnist from './data/fashion_mnist_10000.png'
//import fashion_mnist_labels from './data/fashion_mnist_labels_10000.npy'
//import mnist from './data/mnist.png'
//import mnist_labels from './data/mnist_labels_uint8.dat'
//import fashion_mnist from './data/small/mnist_100.png'
//import fashion_mnist_labels from './data/small/mnist_100_labels.npy'
//import fashion_mnist from './data/small/fashion_mnist_1000.png'
//import fashion_mnist_labels from './data/small/fashion_mnist_1000_labels.npy'

import { Button, Checkbox, DataLoaderOpts, DropdownOpts } from './types'

import './styles.scss'

async function run() {
  const data = new MnistData()
  await data.load()

  const gui = new GUI()

  const dropdowns: DropdownOpts[] = [
    {
      name: 'dataset-choice',
      callback: loadDataset,
    },
  ]

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
    imagesPath: './data/small/mnist_100.png',
    labelsPath: './data/small/mnist_100_labels.png',
    ratio: 6 / 7,
    numClasses: 10,
  })
  await dataLoader.loadImages()

  gui.initDropdown(dropdowns)

  //const model = await new CNN(gui)
  //await model.warm()

  const model = await new Gen(gui, 7 * 7)
  await model.warm()

  const editor = new Editor()
  await gui.initModel(model, editor)
  gui.initOutput(model)
  gui.initChart('loss')

  async function loadDataset() {
    const imagesPath = this.options[this.selectedIndex].dataset.images
    const labelsPath = this.options[this.selectedIndex].dataset.labels

    dataLoader.options = {
      imagesPath: imagesPath,
      labelsPath: labelsPath,
    }
    await dataLoader.loadImages()
  }

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
      await model.train(dataLoader)
    }
  }

  async function predict() {
    const pred = await model.doPrediction()
    tf.browser.toPixels(pred as tf.Tensor3D, gui.output.modelOutput)
  }
}

document.addEventListener('DOMContentLoaded', run)
