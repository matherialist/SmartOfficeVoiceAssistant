# SmartOfficeVoiceAssistant
# REST API
## /get-intent
Type: POST request
```
{"audio": "audio as base64 string"}
```
Output: JSON with command and system response
```
{
  "command": {
    "device": "light",
    "action": "increase",
    "parameter": "brightness"
    "value": null
  },
  "response": "the room is too dark"
}
```
