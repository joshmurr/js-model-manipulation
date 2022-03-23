/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/* eslint-disable @typescript-eslint/no-unused-vars */

import * as tf from '@tensorflow/tfjs'
import NP_Loader from './NPYLoader'
import { TypedArray, DataLoaderOpts, NPYLoaded } from './types'

export default class DataLoader {
  private imagesPath: string
  private labelsPath: string
  private shuffledTrainIndex: number
  private shuffledTestIndex: number
  private trainIndices: Uint32Array
  private testIndices: Uint32Array
  private trainImages: TypedArray
  private testImages: TypedArray
  private trainLabels: TypedArray
  private testLabels: TypedArray
  private datasetImages: TypedArray
  private datasetLabels: TypedArray
  private npyLoader: NP_Loader
  private labelsType: string
  private ratio: number
  private numTrain: number
  private numTest: number
  private IMAGE_SIZE: number
  private NUM_DATASET_ELEMENTS: number
  private NUM_CLASSES: number

  constructor(opts: DataLoaderOpts) {
    this.shuffledTrainIndex = 0
    this.shuffledTestIndex = 0

    this.imagesPath = opts.imagesPath
    this.labelsPath = opts.labelsPath

    // Get file extension
    this.labelsType = this.labelsPath.split('.').slice(-1).pop()

    this.ratio = opts.ratio
    this.NUM_CLASSES = opts.numClasses

    this.npyLoader = new NP_Loader()
  }

  public async loadImages() {
    // Make a request for the MNIST sprited image.
    const img = new Image()
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    const imgRequest = new Promise<void>((resolve, reject) => {
      img.crossOrigin = ''
      img.onload = () => {
        img.width = img.naturalWidth
        img.height = img.naturalHeight

        this.IMAGE_SIZE = img.naturalWidth
        this.NUM_DATASET_ELEMENTS = img.naturalHeight

        const datasetBytesBuffer = new ArrayBuffer(
          this.NUM_DATASET_ELEMENTS * this.IMAGE_SIZE * 4
        )

        const chunkSize = 5000
        canvas.width = img.width
        canvas.height = chunkSize

        for (let i = 0; i < this.NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer,
            i * this.IMAGE_SIZE * chunkSize * 4,
            this.IMAGE_SIZE * chunkSize
          )
          ctx.drawImage(
            img,
            0,
            i * chunkSize,
            img.width,
            chunkSize,
            0,
            0,
            img.width,
            chunkSize
          )

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer)

        resolve()
      }
      img.src = this.imagesPath
    })

    const imgResponse = await Promise.resolve(imgRequest)

    this.numTrain = Math.floor(this.ratio * this.NUM_DATASET_ELEMENTS)
    this.numTest = this.NUM_DATASET_ELEMENTS - this.numTrain

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(this.numTrain)
    this.testIndices = tf.util.createShuffledIndices(this.numTrain)

    // Slice the the images and labels into train and test sets.
    this.trainImages = this.datasetImages.slice(
      0,
      this.IMAGE_SIZE * this.numTrain
    )
    this.testImages = this.datasetImages.slice(this.IMAGE_SIZE * this.numTrain)
  }

  public async loadLabels() {
    switch (this.labelsType) {
      case 'npy':
        {
          const labelsRequest = this.npyLoader.load(this.labelsPath)
          const labelsResponse = await Promise.resolve(labelsRequest)
          this.datasetLabels = labelsResponse.data
        }
        break
      case 'dat':
        {
          const labelsRequest = fetch(this.labelsPath)
          const labelsResponse = await Promise.resolve(labelsRequest)
          this.datasetLabels = new Uint8Array(
            await labelsResponse.arrayBuffer()
          )
        }
        break
    }

    this.trainLabels = this.datasetLabels.slice(
      0,
      this.NUM_CLASSES * this.numTrain
    )
    this.testLabels = this.datasetLabels.slice(this.NUM_CLASSES * this.numTrain)
  }

  nextTrainBatch(batchSize: number) {
    return this.nextBatch(
      batchSize,
      [this.trainImages, this.trainLabels],
      () => {
        this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length
        return this.trainIndices[this.shuffledTrainIndex]
      }
    )
  }

  nextTestBatch(batchSize: number) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
        (this.shuffledTestIndex + 1) % this.testIndices.length
      return this.testIndices[this.shuffledTestIndex]
    })
  }

  single() {
    return this.nextBatch(1, [this.testImages, this.testLabels], () => {
      return this.testIndices[this.shuffledTrainIndex + 1]
    })
  }

  nextBatch(batchSize: number, data: TypedArray[], index: () => number) {
    const batchImagesArray = new Float32Array(batchSize * this.IMAGE_SIZE)
    const batchLabelsArray = new Uint8Array(batchSize * this.NUM_CLASSES)

    for (let i = 0; i < batchSize; i++) {
      const idx = index()

      const image = data[0].slice(
        idx * this.IMAGE_SIZE,
        idx * this.IMAGE_SIZE + this.IMAGE_SIZE
      )
      batchImagesArray.set(image, i * this.IMAGE_SIZE)

      if (data[1] !== undefined) {
        const label = data[1].slice(
          idx * this.NUM_CLASSES,
          idx * this.NUM_CLASSES + this.NUM_CLASSES
        )
        batchLabelsArray.set(label, i * this.NUM_CLASSES)
      }
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, this.IMAGE_SIZE])
    let labels = undefined
    if (data[1] !== undefined) {
      labels = tf.tensor2d(batchLabelsArray, [batchSize, this.NUM_CLASSES])
    }

    return { xs, labels }
  }
}
