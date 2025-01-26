import "math"
import "date"

// 1) Query data
from(bucket: "homeassistant/autogen")
  |> range(start: 2024-10-12T23:00:00Z)

  // 2) Filter relevant sensors
  |> filter(fn: (r) =>
    (r.entity_id == "stufa_salotto_status" and r._measurement == "state" and r._field == "raw_value") or
    (r.entity_id == "average_temperature" and r._measurement == "°C" and r._field == "value") or
    (r.entity_id == "termoigrometro_soggiorno_temperatura" and r._measurement == "°C" and r._field == "value") or
    (r.entity_id == "temperature_comfort" and r._measurement == "°C" and r._field == "value") or
    (r.entity_id == "termoigrometro_soggiorno_umidita" and r._measurement == "%"  and r._field == "value") or
    (r.entity_id == "blink_g8t1_gj00_1403_0p8d_temperatura" and r._measurement == "°C" and r._field == "value") or
    (r.entity_id == "stufa_salotto_power" and r._measurement == "state" and r._field == "value") or
    (r.entity_id == "stufa_salotto_real_power" and r._measurement == "state" and r._field == "value"))

  // 3) Pivot to wide format
  |> pivot(
    rowKey: ["_time"],
    columnKey: ["entity_id"],
    valueColumn: "_value"
  )

  // 4) Merge all groups into one table
  |> group(columns: [])

  // 5) Rename fields and compute derived columns
  |> map(fn: (r) => ({
      _time: r._time,
      stove_status: float(v: r.stufa_salotto_status),
      avg_temperature: float(v: r.average_temperature),
      living_room_temperature: float(v: r.termoigrometro_soggiorno_temperatura),
      setpoint_temperature: float(v: r.temperature_comfort),
      living_room_humidity: float(v: r.termoigrometro_soggiorno_umidita),
      outdoor_temperature: float(v: r.blink_g8t1_gj00_1403_0p8d_temperatura),
      stove_set_power: int(v: r.stufa_salotto_power),
      stove_actual_power: int(v: r.stufa_salotto_real_power),
  }))

  // 6) Keep final columns
  |> keep(columns: [
    "_time",
    "stove_status", "avg_temperature", "living_room_temperature",
    "setpoint_temperature", "living_room_humidity", "outdoor_temperature",
    "stove_set_power", "stove_actual_power",
  ])
