import * as tf from '@tensorflow/tfjs'
import Model from './Model'
import Editor from './Editor'
import Chart from './Chart'
import { Button, Checkbox, PixelData, DecodedKernel } from './types'

export default class GUI {
  private container: HTMLElement
  private display: HTMLElement
  private sidebar: HTMLElement
  private editor: Editor
  private chart: Chart
  private _kernelsToUpdate: Set<string>
  private outputSurfaces: { [key: string]: HTMLCanvasElement }
  private kernelStore: { [key: string]: ImageData[] }
  private checkboxes: { [key: string]: HTMLInputElement }
  private tick = 0

  constructor() {
    this._kernelsToUpdate = new Set<string>()
    this.container = document.getElementById('container')
    this.display = document.createElement('div')
    this.display.classList.add('display')
    this.sidebar = document.createElement('div')
    this.sidebar.classList.add('sidebar')

    this.container.appendChild(this.display)
    this.container.appendChild(this.sidebar)

    this.outputSurfaces = {}
    this.kernelStore = {}
    this.checkboxes = {}
  }

  public initButtons(buttons: Array<Button>) {
    buttons.forEach(({ selector, eventListener, callback }) => {
      const buttonEl = document.querySelector(selector)
      buttonEl.addEventListener(eventListener, callback)
    })
  }

  public initCheckboxes(checkboxes: Array<Checkbox>) {
    checkboxes.forEach(({ name, selector }) => {
      const checkboxEl = document.querySelector(selector)
      this.checkboxes[name] = checkboxEl as HTMLInputElement
    })
  }

  public initChart(metrics: string | string[]) {
    this.chart = new Chart(metrics, this.sidebar)
  }

  async initModel(model: Model, editor: Editor) {
    this.editor = editor

    const { layerFilters, layerNames } = model.layerInfo

    layerNames.forEach((layerName, l_id) => {
      const filterArray = layerFilters[layerName]
      const title = document.createElement('h4')
      title.innerText = layerName
      this.display.appendChild(title)

      filterArray.forEach((filter, f_id) => {
        const row = document.createElement('div')
        row.classList.add('filter-row')
        filter.forEach((kernel, k_id) => {
          const [w, h] = kernel.shape
          const canvas = document.createElement('canvas')
          canvas.width = w
          canvas.height = h
          const kernel_id = this.getKernelId(l_id, f_id, k_id)
          canvas.id = kernel_id

          this.kernelStore[kernel_id] = []

          canvas.addEventListener('click', (e) => {
            model.isTraining = false
            editor.show(e, this.setKernel.bind(this))
          })

          row.appendChild(canvas)
        })
        this.display.appendChild(row)
      })
    })

    await this.update(model)
  }

  public initOutput(model: Model) {
    switch (model.outputType) {
      case 'image':
        this.initImageOutput('modelOutput')
        break
      case 'classification':
        //this.initClassificationOutput(model)
        break
      default:
        break
    }
  }

  private initImageOutput(ref: string) {
    const outputContainer = document.createElement('div')
    outputContainer.classList.add('model-output')

    const canvas = document.createElement('canvas')

    this.outputSurfaces[ref] = canvas

    outputContainer.appendChild(canvas)
    this.sidebar.appendChild(outputContainer)
  }

  public displayImage(image: HTMLCanvasElement, ref: string) {
    if (this.outputSurfaces[ref]) return

    const outputContainer = document.createElement('div')
    outputContainer.classList.add('model-output')

    this.outputSurfaces[ref] = image

    outputContainer.appendChild(image)
    this.sidebar.appendChild(outputContainer)
  }

  public setKernel(kernelId: string, data: PixelData) {
    this._kernelsToUpdate.add(kernelId)

    const canvas = <HTMLCanvasElement>document.getElementById(kernelId)
    const ctx = canvas.getContext('2d')
    ctx.putImageData(data.p, data.x, data.y)
  }

  public async update(model: Model, log?: tf.Logs) {
    if (log) {
      this.chart.update(log)
      this.chart.draw()
    }

    if (this.editor.needsUpdate) {
      this.updateModel(model)
    }

    await this.drawFilters(model)

    this.tick++
  }

  private updateModel(model: Model) {
    this._kernelsToUpdate.forEach((id) => {
      const canvas = <HTMLCanvasElement>document.getElementById(id)
      const ctx = canvas.getContext('2d')
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      model.setKernel(this.decodeKernelId(id), imageData)
    })

    this._kernelsToUpdate.clear()
  }

  async drawFilters(model: Model) {
    const layers = await model.getLayers()
    const layerFilters = await model.getFilters(layers)
    const { layerNames } = model.layerInfo

    layerNames.forEach((layerName, l_id) => {
      const filterArray = layerFilters[layerName]
      filterArray.forEach((filter, f_id) => {
        filter.forEach((kernel, k_id) => {
          const kernel_id = this.getKernelId(l_id, f_id, k_id)
          const canvas = <HTMLCanvasElement>document.getElementById(kernel_id)
          const ctx = canvas.getContext('2d')
          const [w, h] = kernel.shape
          const imageData = new ImageData(w, h)
          const data = kernel.dataSync()

          this.basicCanvasUpdate(imageData, data)

          if (this.tick > 0 && this.checkboxes.diff.checked) {
            const diffImageData = this.diffCanvasUpdate(
              imageData,
              data,
              kernel_id
            )
            ctx.putImageData(diffImageData, 0, 0)
          } else {
            ctx.putImageData(imageData, 0, 0)
          }

          this.kernelStore[kernel_id].push(imageData)
        })
      })
    })
  }

  private basicCanvasUpdate(
    imageData: ImageData,
    data: Float32Array | Int32Array | Uint8Array
  ) {
    const { width: w, height: h } = imageData
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        const ix = (y * w + x) * 4
        const iv = y * w + x
        imageData.data[ix + 0] = Math.floor(127 * (data[iv] + 1))
        imageData.data[ix + 1] = Math.floor(127 * (data[iv] + 1))
        imageData.data[ix + 2] = Math.floor(127 * (data[iv] + 1))
        imageData.data[ix + 3] = 255
      }
    }
  }

  private diffCanvasUpdate(
    imageData: ImageData,
    data: Float32Array | Int32Array | Uint8Array,
    kernel_id: string
  ) {
    const { width: w, height: h } = imageData
    const kernelStore = this.kernelStore[kernel_id]
    const epoch = kernelStore.length
    const prevKernel = kernelStore[epoch - 2]

    const diffImageData = new ImageData(w, h)

    const colourScale = 8

    const getScaled = (d: number) => {
      let r, g, b
      if (d < 0) {
        r = 255 + d * colourScale
        g = 255 + d * colourScale
        b = 255
      } else {
        r = 255
        g = 255 - d * colourScale
        b = 255 - d * colourScale
      }

      return [r, g, b]
    }

    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        const ix = (y * w + x) * 4
        const iv = y * w + x

        const d1 = prevKernel.data[ix + 0] - Math.floor(127 * (data[iv] + 1))
        const [r, g, b] = getScaled(d1)

        diffImageData.data[ix + 0] = r
        diffImageData.data[ix + 1] = g
        diffImageData.data[ix + 2] = b
        diffImageData.data[ix + 3] = 255
      }
    }

    return diffImageData
  }

  private getKernelId(l_id: number, f_id: number, k_id: number): string {
    return `l${l_id}-f${f_id}-k${k_id}`
  }

  private decodeKernelId(kernelId: string): DecodedKernel {
    const splits = kernelId.split('-')
    const layerInfo: DecodedKernel = {
      layer: parseInt(splits[0].slice(1)),
      filter: parseInt(splits[1].slice(1)),
      kernel: parseInt(splits[2].slice(1)),
    }
    return layerInfo
  }

  public get output() {
    return this.outputSurfaces
  }
}
