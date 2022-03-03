export default class StateHandler {
  public play = false

  handlePlay() {
    return () => (this.play = !this.play)
  }
}
