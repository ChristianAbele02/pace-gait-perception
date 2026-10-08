using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InputHandler : MonoBehaviour
{
    private static InputHandler _instance;
    public static InputHandler instance
    {
        get
        {
            if (_instance == null)
            {
                _instance = FindObjectOfType<InputHandler>();
                if (_instance == null)
                {
                    // Lazy instantiation
                    GameObject go = new GameObject("InputHandler");
                    _instance = go.AddComponent<InputHandler>();
                    Debug.Log("InputHandler: Auto-created instance.");
                }
            }
            return _instance;
        }
    }

    void Awake()
    {
        if (_instance == null)
        {
            _instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else if (_instance != this)
        {
            Destroy(gameObject);
        }
    }

    public bool GetStartAction()
    {
        return Input.GetKeyDown(KeyCode.Space);
    }

    public bool GetResponseLeft()
    {
        return Input.GetKeyDown(KeyCode.LeftArrow) || Input.GetMouseButtonDown(0);
    }

    public bool GetResponseRight()
    {
        return Input.GetKeyDown(KeyCode.RightArrow) || Input.GetMouseButtonDown(1);
    }
}
