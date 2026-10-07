import XCTest

@MainActor
final class WorkspaceWorkflowTests: XCTestCase {
    func testCreateModelThroughNormalControls() throws {
        continueAfterFailure = false
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("AgentTrainer-UI-\(UUID().uuidString)")
        let app = XCUIApplication(bundleIdentifier: "com.agenttrainer.AgentTrainer")
        app.launchEnvironment["AGENTTRAINER_VALIDATION_WORKSPACE"] = root.path
        app.launch()
        defer { app.terminate(); try? FileManager.default.removeItem(at: root) }
        XCTAssertTrue(app.staticTexts["AI Models"].firstMatch.waitForExistence(timeout: 10))
        app.staticTexts["AI Models"].firstMatch.click()
        let create = app.buttons["models.create"]
        XCTAssertTrue(create.waitForExistence(timeout: 5))
        create.click()
        let name = app.textFields["models.name"]
        XCTAssertTrue(name.waitForExistence(timeout: 5))
        name.click()
        name.typeKey("a", modifierFlags: .command)
        name.typeText("UI validation agent")
        app.buttons["models.confirmCreate"].click()
        XCTAssertTrue(app.staticTexts["UI validation agent"].firstMatch.waitForExistence(timeout: 5))
        let files = try FileManager.default.contentsOfDirectory(at: root.appendingPathComponent("Models"), includingPropertiesForKeys: nil)
        XCTAssertEqual(files.filter { $0.pathExtension == "json" }.count, 1)
        let data = try Data(contentsOf: files[0])
        let model = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(model["name"] as? String, "UI validation agent")
        XCTAssertNil(model["trainedCheckpoint"])
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = "AI Models — normal create workflow"
        screenshot.lifetime = .keepAlways
        add(screenshot)
    }
}
