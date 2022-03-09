export default class Editor {
  private container: HTMLElement
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D

  constructor() {
    this.buildContainer()
  }

  private buildContainer() {
    this.container = document.createElement('div')
    this.container.id = 'editor'

    this.hideDisplay()

    this.container.addEventListener(
      'click',
      function () {
        this.hideDisplay()
      }.bind(this)
    )

    this.canvas = document.createElement('canvas')
    this.ctx = this.canvas.getContext('2d')
    this.canvas.width = 5
    this.canvas.height = 5
    this.container.appendChild(this.canvas)

    document.body.appendChild(this.container)
  }

  public show(event: MouseEvent) {
    const kernel = <HTMLCanvasElement>event.target
    const ctx = kernel.getContext('2d')
    const imgData = ctx.getImageData(0, 0, kernel.width, kernel.height)

    this.ctx.putImageData(imgData, 0, 0)

    this.showDisplay()
  }

  private showDisplay() {
    this.container.classList.remove('hide')
    this.container.classList.add('show')
  }

  private hideDisplay() {
    this.container.classList.add('hide')
    this.container.classList.remove('show')
  }
}
