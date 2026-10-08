using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Trigger volume attached to one checkpoint. Fires
/// <see cref="CheckpointManager.OnPlayerCheckpoint"/> when the participant reaches it,
/// then removes itself so a checkpoint cannot fire twice. Matching is done on
/// <c>transform.root</c> so any collider in the player hierarchy counts.
/// </summary>
public class CheckpointAction : MonoBehaviour
{
    public CheckpointManager manager;
    public int checkpointIndex;

    private void OnTriggerEnter(Collider other)
    {
        if (manager == null || manager.playerObj == null)
        {
            return;
        }

        // Use transform.root to detect collisions with any part of the player hierarchy
        if (other.transform.root.gameObject == manager.playerObj)
        {
            manager.OnPlayerCheckpoint(checkpointIndex);
            
            if (manager.IsDebugMode)
            {
                // In Debug Mode, keep mesh visible, just disable trigger
                Collider col = GetComponent<Collider>();
                if (col != null) col.enabled = false;
            }
            else
            {
                // Normal Mode: Hide/Disable completely
                gameObject.SetActive(false);
            }
        }
    }
}