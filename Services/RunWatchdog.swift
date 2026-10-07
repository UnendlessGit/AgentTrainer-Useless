import Foundation

/// Independent of the model queue, including while Metal is evaluating a graph.
final class RunWatchdog {
    private let timer: DispatchSourceTimer
    init(executor: AgentInputExecutor, maximumHold: Double, maximumRun: Double) {
        let deadline = ProcessInfo.processInfo.systemUptime + maximumRun
        timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .userInteractive))
        timer.schedule(deadline: .now(), repeating: .milliseconds(100))
        timer.setEventHandler {
            if ProcessInfo.processInfo.systemUptime >= deadline { executor.stop("Run time limit reached.") }
            else if executor.longestHold > maximumHold { executor.stop("A key or button reached the configured hold limit.") }
        }
        timer.resume()
    }
    func stop() { timer.cancel() }
    deinit { timer.cancel() }
}
