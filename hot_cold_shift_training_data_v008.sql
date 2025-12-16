-- Slot-level training dataset for hot/cold prediction (v004)
-- Labels per shift slot based on slot_capacity utilization

CREATE TEMP FUNCTION GetShiftwiseTableCapacity(
  duration_dict_ranges STRING,
  table_size INT64,
  shift_start_time_min INT64,
  shift_end_time_min INT64
)
RETURNS BIGNUMERIC
LANGUAGE js AS """
  try {
    var party_sizes = JSON.parse(duration_dict_ranges);
    var correct_end_time = parseInt(shift_end_time_min);
    var correct_start_time = parseInt(shift_start_time_min);
    var turn_duration = 0;
    var total_turns = 0;
    var parsed_table_size = parseInt(table_size);
    var json_table_size = parsed_table_size;

    if (isNaN(parsed_table_size) || parsed_table_size == null) {
      return null;
    } else {
      if (parsed_table_size > 9) {
        json_table_size = 9;
      }
    }

    if (correct_end_time - correct_start_time < 0) {
      correct_end_time += 24 * 60;
    }

    var keys = Object.keys(party_sizes);
    var current_order = correct_start_time;

    for (var time of keys) {
      var time_parts = time.split(':');
      var range_start_time_min = parseInt(time_parts[0]) * 60 + parseInt(time_parts[1]);
      var index = keys.indexOf(time);
      var next_time_parts = index === keys.length - 1 ? null : keys[index + 1].split(':');
      var range_end_time_min = next_time_parts ? (parseInt(next_time_parts[0]) * 60 + parseInt(next_time_parts[1])) : correct_end_time;

      if (range_end_time_min < range_start_time_min) {
        range_end_time_min += 24 * 60;
      }

      if (correct_start_time >= range_end_time_min || correct_end_time <= range_start_time_min) {
        continue;
      }

      var effective_start = Math.max(current_order, range_start_time_min);
      var effective_end = Math.min(correct_end_time, range_end_time_min);
      turn_duration = party_sizes[time][json_table_size];

      if (turn_duration < 1) {
        return null;
      }

      while (effective_start <= effective_end) {
        var interval_end = effective_start + turn_duration;
        if (next_time_parts != null && effective_end == effective_start){
          break
        }
        if (interval_end > effective_end && effective_end != effective_start && effective_end < effective_start) {
          break;
        }
        total_turns += parsed_table_size;
        effective_start = interval_end;
      }
      current_order = Math.max(current_order, effective_start);
    }
    return isNaN(total_turns) ? null : total_turns;
  } catch (error) {
    return null;
  }
""";

CREATE TEMP FUNCTION GetShiftCapacity(
  custom_pacing_json_as_str STRING,
  covers_per_seating_interval INT64,
  shift_start_time_min INT64,
  shift_end_time_min INT64,
  interval_minutes INT64
) RETURNS BIGNUMERIC LANGUAGE js AS
"""
  try {
    var pacing = 0
    var counter = Number(shift_start_time_min)
    var correct_end_time = parseInt(shift_end_time_min)
    var correct_interval_minutes = parseInt(interval_minutes)

    if (shift_end_time_min - shift_start_time_min < 0){
      correct_end_time += 24 * 60
    }

    var custom_pacing_json = JSON.parse(custom_pacing_json_as_str)

    while (counter <= correct_end_time){
      var wrapped_counter = counter % (24 * 60);
      var temp = custom_pacing_json[wrapped_counter.toString()]
      counter += correct_interval_minutes;

      if (temp >= 0){
        pacing += temp
        continue
      }

      if (covers_per_seating_interval){
        pacing += parseInt(covers_per_seating_interval)
      }
    }
    return pacing
  } catch (e) {
    return null;
  }
""";

-- Helper to read a pacing value at a given minute key from JSON object
CREATE TEMP FUNCTION GetPacingAtMinute(
  custom_pacing_json_as_str STRING,
  minute_key INT64,
  default_val INT64
) RETURNS INT64 LANGUAGE js AS
"""
  try {
    if (!custom_pacing_json_as_str) { return default_val; }
    var obj = JSON.parse(custom_pacing_json_as_str);
    var key = String(minute_key);
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      var v = obj[key];
      if (v === null || v === undefined) { return default_val; }
      var n = parseInt(v);
      if (isNaN(n)) { return default_val; }
      return n;
    }
    return default_val;
  } catch (e) {
    return default_val;
  }
""";

WITH
  ShiftCapacityPerDay AS (
    WITH
      date_series AS (
        SELECT
          TIMESTAMP (DATE) AS DATE,
          (MOD(EXTRACT(DAYOFWEEK FROM DATE) + 5, 7) + 1) AS day_of_week
        FROM
          UNNEST(GENERATE_DATE_ARRAY(DATE_ADD(CURRENT_DATE(), INTERVAL -400 DAY), CURRENT_DATE())) AS DATE
      ),
      shifts_excluded_dates AS (
        SELECT
          shifts.int_id,
          shifts.venue_key_id venue_id,
          excluded_date
        FROM
          `sevenrooms-datawarehouse.sevenrooms_datawarehouse_prod.sr_ShiftDefault_Latest` shifts
        LEFT JOIN
          UNNEST(excluded_dates) AS excluded_date
        WHERE
          excluded_date IS NOT NULL
      ),
      shift_days_of_week AS (
        SELECT
          shifts.int_id,
          shifts.venue_key_id venue_id,
          CONCAT(
            CASE WHEN shifts.day_of_week [SAFE_OFFSET (0)] = TRUE THEN 1 ELSE 0 END,
            CASE WHEN shifts.day_of_week [SAFE_OFFSET (1)] = TRUE THEN 2 ELSE 0 END,
            CASE WHEN shifts.day_of_week [SAFE_OFFSET (2)] = TRUE THEN 3 ELSE 0 END,
            CASE WHEN shifts.day_of_week [SAFE_OFFSET (3)] = TRUE THEN 4 ELSE 0 END,
            CASE WHEN shifts.day_of_week [SAFE_OFFSET (4)] = TRUE THEN 5 ELSE 0 END,
            CASE WHEN shifts.day_of_week [SAFE_OFFSET (5)] = TRUE THEN 6 ELSE 0 END,
            CASE WHEN shifts.day_of_week [SAFE_OFFSET (6)] = TRUE THEN 7 ELSE 0 END
          ) AS days_of_week
        FROM
          `sevenrooms-datawarehouse.sevenrooms_datawarehouse_prod.sr_ShiftDefault_Latest` shifts
      ),
      shift_default_dates_with_gaps AS (
        SELECT
          date_series.DATE,
          date_series.day_of_week,
          shifts.int_id,
          shifts.category AS shift_category
        FROM
          date_series
        LEFT JOIN `sevenrooms-datawarehouse.sevenrooms_datawarehouse_prod.sr_ShiftDefault_Latest` shifts ON 1 = 1
        LEFT JOIN shifts_excluded_dates ON shifts.venue_key_id = shifts_excluded_dates.venue_id AND shifts.int_Id = shifts_excluded_dates.int_id AND date_series.DATE = shifts_excluded_dates.excluded_date
        LEFT JOIN shift_days_of_week ON shifts.venue_key_id = shift_days_of_week.venue_id AND shifts.int_id = shift_days_of_week.int_id
        WHERE
          shifts.venue_key_id = 14301284
          AND shifts.effective_start_date <= date_series.DATE
          AND (shifts.effective_end_date >= date_series.DATE OR shifts.effective_end_date IS NULL)
          AND shifts.archived IS NULL
          AND shifts.deleted IS NULL
          AND shifts_excluded_dates.excluded_date IS NULL
          AND INSTR(shift_days_of_week.days_of_week, CAST(date_series.day_of_week AS string)) > 0
      ),
      shift_override_date AS (
        SELECT
          date_series.DATE,
          date_series.day_of_week,
          shifts.int_id,
          shifts.category AS shift_category
        FROM
          date_series
        LEFT JOIN `sevenrooms-datawarehouse.sevenrooms_datawarehouse_prod.sr_ShiftOverride_Latest` shifts ON shifts.DATE = date_series.DATE
        WHERE
          shifts.venue_key_id = 14301284
          AND shifts.deleted IS NULL
      ),
      shift_by_date AS (
        SELECT * FROM shift_default_dates_with_gaps
        UNION ALL
        SELECT * FROM shift_override_date
      ),
      shift_denorm AS (
        SELECT int_id, venue_key_id, floorplan_int_id, persistent_id, interval_minutes, deleted, name, start_time, end_time, effective_start_date, effective_end_date, table_size, turn_duration, venue_start_of_day_time_min, last_updated, custom_pacing_str, covers_per_seating_interval, shift_start_time_min, shift_end_time_min, duration_minutes_by_party_size_str, duration_dict_ranges_str FROM `sevenrooms-datawarehouse.sevenrooms_datawarehouse_prod.ShiftDefault_capacity_denorm`
        UNION ALL
        SELECT int_id, venue_key_id, floorplan_int_id, persistent_id, interval_minutes, deleted, name, start_time, end_time, effective_start_date, effective_end_date, table_size, turn_duration, venue_start_of_day_time_min, last_updated, custom_pacing_str, covers_per_seating_interval, shift_start_time_min, shift_end_time_min, duration_minutes_by_party_size_str, duration_dict_ranges_str FROM `sevenrooms-datawarehouse.sevenrooms_datawarehouse_prod.ShiftOverride_capacity_denorm`
      ),
      shift_denorm_by_date AS (
        SELECT
          shift_by_date.date,
          shift_by_date.day_of_week,
          shift_by_date.shift_category,
          shift_denorm.*
        FROM
          shift_by_date
        LEFT JOIN shift_denorm ON shift_denorm.int_id = shift_by_date.int_id
      ),
      shift_util AS (
        SELECT
          CASE
            WHEN max_shift_capacity < calculated_capacity THEN max_shift_capacity
            ELSE calculated_capacity
          END AS util_capacity,
          shift_util_temp.*
        FROM (
          SELECT
            shifts.DATE,
            shifts.int_id,
            shifts.name,
            shifts.persistent_id,
            shifts.venue_key_id,
            shifts.shift_category,
            shifts.interval_minutes,
            shifts.custom_pacing_str,
            shifts.covers_per_seating_interval,
            shifts.shift_start_time_min,
            shifts.shift_end_time_min,
            GetShiftCapacity(custom_pacing_str, covers_per_seating_interval, shift_start_time_min, shift_end_time_min, interval_minutes) AS max_shift_capacity,
            SUM(GetShiftwiseTableCapacity(duration_dict_ranges_str, table_size, shift_start_time_min, shift_end_time_min)) AS calculated_capacity
          FROM
            shift_denorm_by_date shifts
          GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        ) shift_util_temp
      )
    SELECT DATE, persistent_id, venue_key_id, util_capacity, shift_category, interval_minutes, custom_pacing_str, covers_per_seating_interval, shift_start_time_min, shift_end_time_min
    FROM shift_util
    WHERE DATE(DATE) < CURRENT_DATE() AND venue_key_id = 14301284
  ),

  -- Build slots per shift
  ShiftSlots AS (
    SELECT
      s.DATE AS shift_date,
      s.venue_key_id,
      s.persistent_id,
      s.shift_category,
      s.util_capacity,
      s.interval_minutes,
      s.custom_pacing_str,
      s.covers_per_seating_interval,
      s.shift_start_time_min,
      s.shift_end_time_min,
      -- slot index starting at 0
      slot_index,
      MOD(s.shift_start_time_min + slot_index * s.interval_minutes, 24*60) AS slot_start_time_min,
      MOD(s.shift_start_time_min + (slot_index + 1) * s.interval_minutes, 24*60) AS slot_end_time_min,
      -- Handle overnight by computing exact timestamps with day offsets
      TIMESTAMP_ADD(
        TIMESTAMP_ADD(
          TIMESTAMP_TRUNC(s.DATE, DAY),
          INTERVAL MOD(s.shift_start_time_min + slot_index * s.interval_minutes, 24*60) MINUTE
        ),
        INTERVAL CAST(DIV(s.shift_start_time_min + slot_index * s.interval_minutes, 24*60) AS INT64) DAY
      ) AS slot_start_timestamp,
      TIMESTAMP_ADD(
        TIMESTAMP_ADD(
          TIMESTAMP_TRUNC(s.DATE, DAY),
          INTERVAL MOD(s.shift_start_time_min + (slot_index + 1) * s.interval_minutes, 24*60) MINUTE
        ),
        INTERVAL CAST(DIV(s.shift_start_time_min + (slot_index + 1) * s.interval_minutes, 24*60) AS INT64) DAY
      ) AS slot_end_timestamp
    FROM ShiftCapacityPerDay s,
    UNNEST(GENERATE_ARRAY(0,
      CAST( (
        CASE
          WHEN s.shift_end_time_min >= s.shift_start_time_min THEN (s.shift_end_time_min - s.shift_start_time_min)
          ELSE (s.shift_end_time_min + 24*60 - s.shift_start_time_min)
        END) / s.interval_minutes AS INT64) - 1)) AS slot_index
  ),

  -- Compute pacing weights per slot from custom pacing or fallback
  SlotWeights AS (
    SELECT
      ss.*,
      GetPacingAtMinute(s.custom_pacing_str, MOD(ss.slot_start_time_min, 24*60), NULL) AS raw_pacing,
      COALESCE(GetPacingAtMinute(s.custom_pacing_str, MOD(ss.slot_start_time_min, 24*60), NULL), s.covers_per_seating_interval, 1) AS pacing_weight
    FROM ShiftSlots ss
    JOIN ShiftCapacityPerDay s
      ON s.DATE = ss.shift_date AND s.venue_key_id = ss.venue_key_id AND s.persistent_id = ss.persistent_id
  ),

  SlotWeightsNorm AS (
    SELECT
      sw.*,
      SUM(pacing_weight) OVER (PARTITION BY shift_date, venue_key_id, persistent_id) AS total_weight
    FROM SlotWeights sw
  ),

  SlotCapacity AS (
    SELECT
      swn.*,
      CASE
        WHEN total_weight IS NULL OR total_weight = 0 THEN SAFE_DIVIDE(util_capacity, NULLIF(COUNT(*) OVER (PARTITION BY shift_date, venue_key_id, persistent_id), 0))
        ELSE util_capacity * (SAFE_DIVIDE(pacing_weight, total_weight))
      END AS slot_capacity
    FROM SlotWeightsNorm swn
  ),

  -- Prefilter reservations to reduce scanned data
  ReservationsFiltered AS (
    SELECT *
    FROM `sevenrooms-datawarehouse.sevenrooms_datawarehouse_prod.nightloop_ReservationActual_Latest`
    WHERE venue_key_id = 14301284
      AND status NOT IN ('CANCELED')
      AND marked_time >= TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL -1095 DAY)
      AND marked_time <= CURRENT_TIMESTAMP()
  ),

  -- Reservations aggregated into slots (final actuals)
  SlotReservations AS (
    SELECT
      sc.shift_date,
      sc.venue_key_id,
      sc.persistent_id,
      sc.shift_category,
      sc.slot_index,
      SUM(ra.max_guests) AS covers_final_in_slot
    FROM SlotCapacity sc
    LEFT JOIN ReservationsFiltered ra
      ON ra.venue_key_id = sc.venue_key_id
      AND ra.marked_time >= sc.slot_start_timestamp
      AND ra.marked_time < sc.slot_end_timestamp
    GROUP BY 1,2,3,4,5
  )

SELECT
  sc.shift_date,
  sc.venue_key_id,
  sc.persistent_id,
  sc.shift_category,
  sc.util_capacity,
  sc.interval_minutes,
  sc.slot_index,
  sc.slot_start_time_min,
  sc.slot_start_timestamp,
  sc.slot_capacity,
  EXTRACT(DAYOFWEEK FROM sc.shift_date) AS day_of_week,
  EXTRACT(MONTH FROM sc.shift_date) AS month,
  EXTRACT(YEAR FROM sc.shift_date) AS year,
  0 AS day_prior,
  COALESCE(sr.covers_final_in_slot, 0) AS covers_as_of_date,
  CASE
    WHEN sc.slot_capacity > 0 AND SAFE_DIVIDE(COALESCE(sr.covers_final_in_slot, 0), sc.slot_capacity) > 0.85 THEN 1
    ELSE 0
  END AS is_hot
FROM SlotCapacity sc
LEFT JOIN SlotReservations sr
  ON sr.shift_date = sc.shift_date
  AND sr.venue_key_id = sc.venue_key_id
  AND sr.persistent_id = sc.persistent_id
  AND sr.slot_index = sc.slot_index
WHERE sc.util_capacity IS NOT NULL AND sc.util_capacity > 0
ORDER BY sc.shift_date, sc.persistent_id, sc.slot_index;


