# Home Assistant Add-on: Stove Heating AI

## How to use

1. Install the add-on
2. Configure the add-on
3. Start the add-on
4. Check the logs of the add-on to see if everything is working correctly
5. Use the rest commands (see below) or the UI to train the AI and to get predictions

## Rest commands to use the AI

`rest_commands.yaml`

```yaml
stove_ai_train:
  url: http://3bd0be80-stove-heating-ai:8099/train
  method: post

stove_ai_status:
  url: http://3bd0be80-stove-heating-ai:8099/status
  method: get

stove_ai_predict:
  url: http://3bd0be80-stove-heating-ai:8099/predict
  method: post
  content_type: "application/json"
  payload: |
    {
      "setpoint_temperature": "{{ setpoint_temperature }}",
      "avg_temperature": "{{ avg_temperature }}",
      "living_room_humidity": "{{ living_room_humidity }}",
      "living_room_temperature": "{{ living_room_temperature }}",
      "outdoor_temperature": "{{ outdoor_temperature }}",
      "stove_set_power": "{{ stove_set_power }}",
      "stove_actual_power": "{{ stove_actual_power }}",
      "time_since_on": "{{ time_since_on }}"
    }
```
