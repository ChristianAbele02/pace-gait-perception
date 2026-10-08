//MIT License
//Copyright 2016-Present
//Ross Tredinnick
//Brady Boettcher
//Living Environments Laboratory
//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
//to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute,
//sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
//INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


using UnityEngine;
using System.Collections;
using UnityEngine.SceneManagement;

public class NetworkingSync : MonoBehaviour {

    private static NetworkingSync instance;

    public string headNodeIP = "127.0.0.1";
    public int port = 7568;
    public int numSlaveNodes = 6;
    public float networkUpdatesPerSecond = 60.0f;
    private string machineName;
    private float myTimeScale = 1.0f;
    private float lastTime = 0.0f;
    private float headTime = 0.0f;
    private bool syncedRandomSeed = false;
    private bool actuallyConnected = false;
    private int connectionTries = 0;
    private const int MAX_CONNECTION_TRIES = 50;
    private int frameCount = 0;

	void Start () {
        if (instance == null)
        {
            instance = this;
            DontDestroyOnLoad(gameObject);

            machineName = System.Environment.MachineName;
            SceneManager.sceneLoaded += OnSceneLoaded;

            if (machineName == MasterTrackingData.HeadNodeMachineName)
            {
			    Debug.Log ("Trying to initialize server: " + numSlaveNodes + ", " + port);
                Network.InitializeServer(numSlaveNodes, port, false);
            }
            else
            {
                Network.Connect(headNodeIP, port);
                connectionTries++;
		    }

            Time.fixedDeltaTime = 0.05f;
            Network.sendRate = networkUpdatesPerSecond;
            SetupNetworkViews();
        }
        else
        {
            Debug.Log("NetworkingSync: Duplicate instance found, destroying...");
            Destroy(this);
        }
	}

    void OnDisable()
    {
        if (instance == this)
            SceneManager.sceneLoaded -= OnSceneLoaded;
    }

    void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        Debug.Log("NetworkingSync: Scene " + scene.name + " loaded. Re-scanning for Rigidbodies...");
        SetupNetworkViews();
    }

    void SetupNetworkViews()
    {
        Rigidbody[] gO = GameObject.FindObjectsOfType<Rigidbody>();
        foreach (Rigidbody rb in gO)
        {
            GameObject gameObject = rb.gameObject;
            if (gameObject.GetComponent<NetworkView>() == null)
            {
                gameObject.AddComponent<NetworkView>();
                NetworkView nv = gameObject.GetComponent<NetworkView>();
                nv.observed = rb;
                nv.stateSynchronization = NetworkStateSynchronization.Unreliable;
            }
        }
    }

    void OnServerInitialized()
    {
        Debug.Log("Server initialized and ready on " + System.Environment.MachineName);
    }

    void OnConnectedToServer()
    {
        Debug.Log(System.Environment.MachineName + " successfully connected to " + MasterTrackingData.HeadNodeMachineName);
        actuallyConnected = true;
    }

    void OnFailedToConnect(NetworkConnectionError error)
    {
        Debug.Log("Could not connect to server on " + MasterTrackingData.HeadNodeMachineName + ": " + error);
        machineName = System.Environment.MachineName;
        if (machineName != MasterTrackingData.HeadNodeMachineName && connectionTries < MAX_CONNECTION_TRIES && !actuallyConnected)
        {
            connectionTries++;
            Debug.Log("Retrying connection... attempt # " + connectionTries);
            Network.Connect(headNodeIP, port);
        }
    }

    void OnDisconnectedFromServer(NetworkDisconnection info)
    {
        if (Network.isServer)
            Debug.Log("Local server connection disconnected");
        else if (info == NetworkDisconnection.LostConnection)
            Debug.Log("Lost connection to the server");
        else
            Debug.Log("Successfully diconnected from the server");
    }

    void Update()
    {
        if (System.Environment.MachineName == MasterTrackingData.HeadNodeMachineName)
        {
            if (Input.inputString.Length > 0)
                GetComponent<NetworkView>().RPC("sendKeys", RPCMode.Others, Input.inputString);

            if (Input.GetKeyDown(KeyCode.Escape))
            {
                GetComponent<NetworkView>().RPC("quitApplication", RPCMode.Others);
                Application.Quit();
            }

            // [ / ] adjust parallax (eye separation) scale across all nodes.
            float parDir = 0f;
            if (Input.GetKeyDown(KeyCode.LeftBracket))       parDir = -0.05f;
            else if (Input.GetKeyDown(KeyCode.RightBracket)) parDir = +0.05f;
            if (parDir != 0f)
                GetComponent<NetworkView>().RPC("ReceiveAdjustParallaxScale", RPCMode.All, parDir);

            // Projection model selection / live-tune across all nodes:
            //   F1 OffAxis (default) | F2 Hybrid | F3 HybridWide | F4/F5/F6 Partial(.3/.5/.7)
            //   F7 HybridSeamless | P toggle OffAxis<->Hybrid
            //   = / -  tune the active param: OffAxis -> swayGain, HybridWide -> widen,
            //          Partial -> blend
            //   , / .  tune the OffAxis render-eye sway smoothing cutoff (Hz)
            //   R      recapture hybrid dRef from the current head pose
            // NOTE: main-row = / - and , / . ONLY (NOT keypad +/- -- SplineWalker uses
            // keypad +/- to nudge walking speed).
            int newMode = -1; float preset = -1f;
            if (Input.GetKeyDown(KeyCode.F1))      { newMode = 0; }
            else if (Input.GetKeyDown(KeyCode.F2)) { newMode = 1; }
            else if (Input.GetKeyDown(KeyCode.F3)) { newMode = 2; }
            else if (Input.GetKeyDown(KeyCode.F4)) { newMode = 3; preset = 0.3f; }
            else if (Input.GetKeyDown(KeyCode.F5)) { newMode = 3; preset = 0.5f; }
            else if (Input.GetKeyDown(KeyCode.F6)) { newMode = 3; preset = 0.7f; }
            else if (Input.GetKeyDown(KeyCode.F7)) { newMode = 4; }
            if (newMode >= 0)
                GetComponent<NetworkView>().RPC("ReceiveSetProjectionMode", RPCMode.All, newMode, preset);

            if (Input.GetKeyDown(KeyCode.P))
                GetComponent<NetworkView>().RPC("ReceiveToggleProjectionMode", RPCMode.All);

            int paramDir = 0;
            if (Input.GetKeyDown(KeyCode.Equals) || Input.GetKeyDown(KeyCode.Plus)) paramDir = +1;
            else if (Input.GetKeyDown(KeyCode.Minus)) paramDir = -1;
            if (paramDir != 0)
                GetComponent<NetworkView>().RPC("ReceiveAdjustParam", RPCMode.All, paramDir);

            // , / .  tune the OffAxis render-eye sway smoothing cutoff (swaySmoothingHz).
            int smoothDir = 0;
            if (Input.GetKeyDown(KeyCode.Period))     smoothDir = +1;
            else if (Input.GetKeyDown(KeyCode.Comma)) smoothDir = -1;
            if (smoothDir != 0)
                GetComponent<NetworkView>().RPC("ReceiveAdjustSwaySmoothing", RPCMode.All, smoothDir);

            if (Input.GetKeyDown(KeyCode.R))
                BroadcastRecalibrateHybrid();

            if (!syncedRandomSeed && frameCount > 500)
            {
                if (Network.connections.Length == numSlaveNodes)
                {
                    GetComponent<NetworkView>().RPC("syncRandomSeed", RPCMode.Others, UnityEngine.Random.seed);
                    syncedRandomSeed = true;
                }
            }

            float t = Time.time;
            if (t - lastTime > 0.1)
            {
                GetComponent<NetworkView>().RPC("getTimeFromHeadnode", RPCMode.Others, t);
                lastTime = t;
            }

            frameCount++;
        }
    }

    [RPC]
    void quitApplication()
    {
        Application.Quit();
    }

    [RPC]
    void ReceiveAdjustParallaxScale(float delta)
    {
        ProjectionPlane.parallaxScale = Mathf.Clamp(
            ProjectionPlane.parallaxScale + delta, 0f, 2f);
        Debug.Log("[Parallax] parallaxScale = " + ProjectionPlane.parallaxScale.ToString("F2"));
    }

    [RPC]
    void ReceiveSetProjectionMode(int modeIndex, float presetParam)
    {
        ProjectionPlane.mode = (ProjectionPlane.ProjectionMode)modeIndex;
        if (ProjectionPlane.mode == ProjectionPlane.ProjectionMode.Partial && presetParam > 0f)
            ProjectionPlane.partialBlend = presetParam;
        Debug.Log("[ProjectionMode] Set to " + ProjectionPlane.mode
            + " (widen=" + ProjectionPlane.hybridWidenFactor.ToString("F2")
            + ", blend=" + ProjectionPlane.partialBlend.ToString("F2") + ")");
    }

    [RPC]
    void ReceiveToggleProjectionMode()
    {
        ProjectionPlane.useStandardPerspective = !ProjectionPlane.useStandardPerspective;
        Debug.Log("[ProjectionMode] Toggled to " + ProjectionPlane.mode);
    }

    [RPC]
    void ReceiveAdjustParam(int direction)
    {
        float delta = 0.05f * direction;
        if (ProjectionPlane.mode == ProjectionPlane.ProjectionMode.HybridWide)
        {
            ProjectionPlane.hybridWidenFactor = Mathf.Clamp(
                ProjectionPlane.hybridWidenFactor + delta, 1.0f, 2.0f);
            Debug.Log("[ProjectionMode] hybridWidenFactor = "
                + ProjectionPlane.hybridWidenFactor.ToString("F2"));
        }
        else if (ProjectionPlane.mode == ProjectionPlane.ProjectionMode.Partial)
        {
            ProjectionPlane.partialBlend = Mathf.Clamp(
                ProjectionPlane.partialBlend + delta, 0.0f, 1.0f);
            Debug.Log("[ProjectionMode] partialBlend = "
                + ProjectionPlane.partialBlend.ToString("F2"));
        }
        else if (ProjectionPlane.mode == ProjectionPlane.ProjectionMode.OffAxis)
        {
            ProjectionPlane.swayGain = Mathf.Clamp01(ProjectionPlane.swayGain + delta);
            Debug.Log("[ProjectionMode] swayGain = " + ProjectionPlane.swayGain.ToString("F2"));
        }
    }

    [RPC]
    void ReceiveAdjustSwaySmoothing(int direction)
    {
        ProjectionPlane.swaySmoothingHz = Mathf.Clamp(
            ProjectionPlane.swaySmoothingHz + 0.2f * direction, 0.2f, 5.0f);
        Debug.Log("[ProjectionMode] swaySmoothingHz = "
            + ProjectionPlane.swaySmoothingHz.ToString("F2"));
    }

    [RPC]
    void ReceiveRecalibrateHybrid()
    {
        ProjectionPlane[] all = GameObject.FindObjectsOfType<ProjectionPlane>();
        foreach (ProjectionPlane pp in all) pp.RecalibrateHybridDRef();
        Debug.Log("[ProjectionMode] Recalibrated Hybrid dRef on " + all.Length + " planes");
    }

    // Re-capture the Hybrid projection reference distance on every node. Called by the
    // R key and by CheckpointManager during the start "stand still" baseline so dRef is
    // locked to the participant's settled start pose. Head-node only (broadcasts to all).
    public void BroadcastRecalibrateHybrid()
    {
        if (System.Environment.MachineName == MasterTrackingData.HeadNodeMachineName)
            GetComponent<NetworkView>().RPC("ReceiveRecalibrateHybrid", RPCMode.All);
    }

    [RPC]
    void sendKeys(string keysPressed)
    {
        Debug.LogError(keysPressed);
    }

    [RPC]
    void syncRandomSeed(int seed)
    {
        Debug.LogError("Synced random seed to " + seed);
        ParticleSystem[] gO = GameObject.FindObjectsOfType<ParticleSystem>();
        foreach (ParticleSystem ps in gO)
            ps.Stop();

        UnityEngine.Random.InitState(seed);

        foreach (ParticleSystem ps in gO)
        {
            ps.useAutoRandomSeed = false;
            ps.randomSeed = (uint)seed;
            ps.Simulate(0.0f, true, true, false);
            ps.Play();
        }
    }

    [RPC]
    void getTimeFromHeadnode(float globalTime)
    {
        if (lastTime == 0.0f)
        {
            lastTime = globalTime;
        }
        else
        {
            float ourTime = Time.time;
            float timeDiff = globalTime - lastTime;
            float scale = ((globalTime - ourTime) + timeDiff) / timeDiff;

            lastTime = globalTime;

            if (scale < 0.0f)
                scale = 0.001f;
            else if (scale > 100.0f)
                scale = 100.0f;

            Time.timeScale = scale;
        }
    }

    public void BroadcastExperimentData(int participantID, int blockID, int trialID, float speed, string sceneName, bool isDebugMode)
    {
        if (System.Environment.MachineName == MasterTrackingData.HeadNodeMachineName)
        {
            Debug.Log("Broadcasting experiment data to all clients...");
            GetComponent<NetworkView>().RPC("ReceiveExperimentData", RPCMode.All, participantID, blockID, trialID, speed, sceneName, isDebugMode);
        }
    }

    [RPC]
    void ReceiveExperimentData(int participantID, int blockID, int trialID, float speed, string sceneName, bool isDebugMode)
    {
        Debug.Log("Received experiment data: P: " + participantID + ", B:" + blockID + ", T:" + trialID + ", Debug:" + isDebugMode);

        if (ExperimentConductor.instance != null)
        {
            ExperimentConductor.instance.participant = participantID;
            ExperimentConductor.instance.block = blockID;
            ExperimentConductor.instance.trial = trialID;
            ExperimentConductor.instance.isDebugMode = isDebugMode;
            ExperimentConductor.instance.experimentSpeed = speed;
        }
        else
        {
            Debug.LogError("Could not find ExperimentConductor instance to save data!");
        }

        UnityEngine.SceneManagement.SceneManager.LoadScene(sceneName);
    }

    [RPC]
    void ReceiveStartGame()
    {
        Debug.Log("NetworkingSync: Received StartGame RPC.");

        if (CheckpointManager.instance != null)
            CheckpointManager.instance.StartGame();
        else
            Debug.LogError("NetworkingSync: Received StartGame RPC, but CheckpointManager.instance is null!");
    }

    public void BroadcastStartGame()
    {
        if (System.Environment.MachineName == MasterTrackingData.HeadNodeMachineName)
            GetComponent<NetworkView>().RPC("ReceiveStartGame", RPCMode.All);
    }

    [RPC]
    void ReceiveShowTask(int direction, float angle)
    {
        if (FollowerPlane.instance != null)
            FollowerPlane.instance.ShowTask(direction, angle);
        else
            Debug.LogError("NetworkSync: Received ShowTask RPC, but FollowerPlane.instance is null.");
    }

    public void BroadcastShowTask(int direction, float angle)
    {
        if (System.Environment.MachineName == MasterTrackingData.HeadNodeMachineName)
            GetComponent<NetworkView>().RPC("ReceiveShowTask", RPCMode.All, direction, angle);
    }

    [RPC]
    void ReceiveHideCheckpoint(int checkpointIndex)
    {
        if (CheckpointManager.instance != null)
            CheckpointManager.instance.SetCheckpointVisible(checkpointIndex, false);
    }

    public void BroadcastHideCheckpoint(int checkpointIndex)
    {
        if (System.Environment.MachineName == MasterTrackingData.HeadNodeMachineName)
            GetComponent<NetworkView>().RPC("ReceiveHideCheckpoint", RPCMode.Others, checkpointIndex);
    }

    [RPC]
    void ReceiveResetTask()
    {
        if (FollowerPlane.instance != null)
            FollowerPlane.instance.ResetTask();
        else
            Debug.LogError("NetworkSync: Received ResetTask RPC, but FollowerPlane.instance is null.");
    }

    public void BroadcastResetTask()
    {
        if (System.Environment.MachineName == MasterTrackingData.HeadNodeMachineName)
            GetComponent<NetworkView>().RPC("ReceiveResetTask", RPCMode.All);
    }

    [RPC]
    void ReceiveResponseReceived(int response)
    {
        Debug.Log("NetworkingSync: Received Response: " + response);
        if (CheckpointManager.instance != null)
            CheckpointManager.instance.OnResponseReceived(response);
    }

    public void BroadcastResponseReceived(int response)
    {
        if (System.Environment.MachineName == MasterTrackingData.HeadNodeMachineName)
            GetComponent<NetworkView>().RPC("ReceiveResponseReceived", RPCMode.All, response);
    }

    [RPC]
    void ReceiveShowMessage(string message)
    {
        HeadMessageDisplay.Instance.ShowMessage(message);
    }

    // Show a head-height text message (baseline / treadmill warning / cool-down) on
    // every node. RPCMode.All so the head node displays it too.
    public void BroadcastShowMessage(string message)
    {
        if (System.Environment.MachineName == MasterTrackingData.HeadNodeMachineName)
            GetComponent<NetworkView>().RPC("ReceiveShowMessage", RPCMode.All, message);
    }

    [RPC]
    void ReceiveHideMessage()
    {
        HeadMessageDisplay.Instance.Hide();
    }

    public void BroadcastHideMessage()
    {
        if (System.Environment.MachineName == MasterTrackingData.HeadNodeMachineName)
            GetComponent<NetworkView>().RPC("ReceiveHideMessage", RPCMode.All);
    }

    [RPC]
    void ReceiveReturnToScene(int sceneIndex)
    {
        Debug.Log("NetworkingSync: Received command to load scene index " + sceneIndex);
        UnityEngine.SceneManagement.SceneManager.LoadScene(sceneIndex);
    }

    public void BroadcastReturnToScene(int sceneIndex)
    {
        if (System.Environment.MachineName == MasterTrackingData.HeadNodeMachineName)
            GetComponent<NetworkView>().RPC("ReceiveReturnToScene", RPCMode.All, sceneIndex);
    }
}
