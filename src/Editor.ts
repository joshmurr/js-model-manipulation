import { DrawCallback } from './types'

export default class Editor {
  private container: HTMLElement
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private tools: HTMLElement
  private SHIFT = false
  private SCALE = 30
  private _needsUpdate = false
  private setKernelCB: DrawCallback
  private currentKernel: string

  constructor() {
    this.buildContainer()
  }

  private buildContainer() {
    this.container = document.createElement('div')
    this.container.id = 'editor'

    this.hideDisplay()

    this.tools = document.createElement('div')

    const buttons = [
      {
        text: 'close',
        parent: this.container,
        callback: () => this.hideDisplay(),
      },
      {
        text: 'black',
        parent: this.tools,
        callback: () => this.fill('black'),
      },
      {
        text: 'white',
        parent: this.tools,
        callback: () => this.fill('white'),
      },
      {
        text: 'grey',
        parent: this.tools,
        callback: () => this.fill('grey'),
      },
    ]

    buttons.forEach(({ text, parent, callback }) =>
      this.addButton(text, parent, callback)
    )

    this.canvas = document.createElement('canvas')
    this.ctx = this.canvas.getContext('2d')

    this.container.appendChild(this.canvas)
    this.container.appendChild(this.tools)
    document.body.appendChild(this.container)
  }

  public show(event: MouseEvent, setKernelCB: DrawCallback) {
    this.setKernelCB = setKernelCB

    const kernel = <HTMLCanvasElement>event.target
    this.canvas.width = kernel.width
    this.canvas.height = kernel.height
    this.canvas.style.width = `${kernel.width * this.SCALE}px`
    this.canvas.style.height = `${kernel.height * this.SCALE}px`

    const ctx = kernel.getContext('2d')
    const imgData = ctx.getImageData(0, 0, kernel.width, kernel.height)

    this.currentKernel = kernel.id

    this.ctx.putImageData(imgData, 0, 0)
    this.showDisplay()
  }

  private addButton(
    text: string,
    parent: HTMLElement,
    callback: (e?: MouseEvent) => void
  ) {
    const button = document.createElement('span')
    button.innerText = text
    button.addEventListener('click', callback)
    parent.appendChild(button)
  }

  private handleKeyDown(e: KeyboardEvent) {
    if (e.shiftKey) this.SHIFT = true
  }

  private handleKeyUp() {
    this.SHIFT = false
  }

  private showDisplay() {
    this.container.classList.remove('hide')
    this.container.classList.add('show')
    this.canvas.addEventListener('click', (e) => {
      this.draw(e)
    })

    document.addEventListener('keydown', (e) => {
      this.handleKeyDown(e)
    })
    document.addEventListener('keyup', () => {
      this.handleKeyUp()
    })
  }

  private hideDisplay() {
    this.container.classList.add('hide')
    this.container.classList.remove('show')
    if (this.canvas) {
      this.canvas.removeEventListener('click', this.draw)
    }
    document.removeEventListener('keydown', this.handleKeyDown)
    document.removeEventListener('keyup', this.handleKeyUp)
  }

  public get needsUpdate(): boolean {
    const currentStatus = this._needsUpdate
    this._needsUpdate = false
    return currentStatus
  }

  private text2Colour(colour: string): number {
    let fillColour: number
    switch (colour) {
      case 'black':
        fillColour = 0
        break
      case 'white':
        fillColour = 255
        break
      case 'grey':
      default:
        fillColour = 128
        break
    }
    return fillColour
  }

  private draw(event: MouseEvent) {
    const rect = this.canvas.getBoundingClientRect()
    const x = Math.floor((event.clientX - rect.left) / this.SCALE)
    const y = Math.floor((event.clientY - rect.top) / this.SCALE)
    const p = this.ctx.getImageData(x, y, 1, 1)
    const data = p.data
    const adder = this.SHIFT ? 10 : -10
    data[0] += adder
    data[1] += adder
    data[2] += adder
    this.ctx.putImageData(p, x, y)
    if (this.setKernelCB) {
      this._needsUpdate = true
      this.setKernelCB(this.currentKernel, { p, x, y })
    }
  }

  private fill(colour: string) {
    const { width, height } = this.canvas
    const imageData = this.ctx.getImageData(0, 0, width, height)
    const fillColour = this.text2Colour(colour)
    const data = imageData.data.map((c, i) =>
      (i + 1) % 4 === 0 ? c : fillColour
    )
    const newImageData = new ImageData(data, width, height)
    this.ctx.putImageData(newImageData, 0, 0)
    if (this.setKernelCB) {
      this._needsUpdate = true
      this.setKernelCB(this.currentKernel, { p: newImageData, x: 0, y: 0 })
    }
  }
}
