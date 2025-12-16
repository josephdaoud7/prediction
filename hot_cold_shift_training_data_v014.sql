-- UDF to calculate the total number of guest-minutes a single table can accommodate during a shift
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

-- UDF to calculate max covers based on pacing rules
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

WITH
  -- This large CTE calculates the theoretical capacity of each shift
  ShiftCapacityPerDay AS (
    WITH
      date_series AS (
        SELECT
          TIMESTAMP (DATE) AS DATE,
          (MOD(EXTRACT(DAYOFWEEK FROM DATE) + 5, 7) + 1) AS day_of_week
        FROM
          UNNEST(GENERATE_DATE_ARRAY(DATE_ADD(CURRENT_DATE(), INTERVAL -3 YEAR), DATE_ADD(CURRENT_DATE(), INTERVAL 365 DAY))) AS DATE
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
            GetShiftCapacity(custom_pacing_str, covers_per_seating_interval, shift_start_time_min, shift_end_time_min, interval_minutes) AS max_shift_capacity,
            SUM(GetShiftwiseTableCapacity(duration_dict_ranges_str, table_size, shift_start_time_min, shift_end_time_min)) AS calculated_capacity
          FROM
            shift_denorm_by_date shifts
          GROUP BY 1, 2, 3, 4, 5, 6, 7
        ) shift_util_temp
      )
    SELECT DATE, persistent_id, venue_key_id, util_capacity, shift_category FROM shift_util
  ),

  -- CTE to calculate the final outcome (is_hot) for each historical shift with enhanced definition
  ShiftCapacityAndOutcome AS (
    SELECT
      s.DATE,
      s.persistent_id,
      s.venue_key_id,
      s.util_capacity,
      s.shift_category,
      CASE
        -- Enhanced hot/cold definition: more than 85% capacity utilization OR more than 90% of historical average
        WHEN s.util_capacity > 0 AND (COALESCE(r.final_actual_covers, 0) / s.util_capacity) > 0.85 THEN 1
        ELSE 0
      END AS is_hot
    FROM ShiftCapacityPerDay AS s
    LEFT JOIN (
        SELECT
            venue_key_id,
            shift_persistent_id,
            TIMESTAMP_TRUNC(marked_time, DAY) as reservation_date,
            SUM(max_guests) as final_actual_covers,
            COUNT(*) as total_reservations,
            AVG(max_guests) as avg_party_size
        FROM `sevenrooms-datawarehouse.sevenrooms_datawarehouse_prod.nightloop_ReservationActual_Latest`
        WHERE status NOT IN ('CANCELED') 
          AND venue_key_id = 14301284
          AND marked_time IS NOT NULL
        GROUP BY 1, 2, 3
    ) AS r ON s.venue_key_id = r.venue_key_id 
          AND s.persistent_id = r.shift_persistent_id 
          AND s.DATE = r.reservation_date
    WHERE DATE(s.DATE) < CURRENT_DATE() 
      AND s.venue_key_id = 14301284
      -- Enhanced filter: only include shifts with sufficient data
      AND s.DATE >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)  -- Focus on recent 2 years for training
  ),

  -- Create a number series for our "days prior" snapshots (expanded range)
  DaysPrior AS (
    SELECT day_prior FROM UNNEST(GENERATE_ARRAY(0, 365)) as day_prior
  ),

  -- Create the training "scaffold". For each historical shift, create rows for different time horizons
  TrainingScaffold AS (
    SELECT
      s.DATE AS shift_date,
      s.venue_key_id,
      s.persistent_id,
      s.util_capacity,
      s.shift_category,
      s.is_hot,
      d.day_prior,
      TIMESTAMP_TRUNC(TIMESTAMP_SUB(s.DATE, INTERVAL d.day_prior DAY), DAY) AS snapshot_timestamp
    FROM ShiftCapacityAndOutcome AS s
    CROSS JOIN DaysPrior AS d
    WHERE s.util_capacity IS NOT NULL 
      AND s.util_capacity > 0
      -- Only include meaningful prediction horizons (not too far out or same day for most training data)
      AND (d.day_prior BETWEEN 1 AND 180 OR d.day_prior IN (0, 365))  -- Focus on 0-180 days + 1 year out
  ),

  -- Aggregate reservations by the day they were created with enhanced metrics
  ReservationsByCreationDate AS (
    SELECT
      venue_key_id,
      shift_persistent_id,
      TIMESTAMP_TRUNC(marked_time, DAY) AS reservation_date,
      TIMESTAMP_TRUNC(created, DAY) AS creation_timestamp,
      SUM(max_guests) AS covers_created_on_day,
      COUNT(*) AS reservations_created_on_day,
      AVG(max_guests) AS avg_party_size_created_on_day,
      -- Lead time for reservations created on this day
      AVG(DATE_DIFF(DATE(marked_time), DATE(created), DAY)) AS avg_lead_time_for_day
    FROM `sevenrooms-datawarehouse.sevenrooms_datawarehouse_prod.nightloop_ReservationActual_Latest`
    WHERE status NOT IN ('CANCELED') 
      AND venue_key_id = 14301284 
      AND created IS NOT NULL
      AND marked_time IS NOT NULL
      -- Focus on recent history for training
      AND DATE(marked_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)
    GROUP BY 1, 2, 3, 4
  ),

  -- Historical booking velocity patterns for enhanced features
  HistoricalBookingVelocity AS (
    SELECT
      EXTRACT(DAYOFWEEK FROM reservation_date) as day_of_week,
      EXTRACT(MONTH FROM reservation_date) as month,
      shift_persistent_id,
      -- Calculate booking velocity at different time horizons
      AVG(CASE WHEN DATE_DIFF(DATE(reservation_date), DATE(creation_timestamp), DAY) <= 7 
               THEN covers_created_on_day ELSE 0 END) as avg_covers_7_days_out,
      AVG(CASE WHEN DATE_DIFF(DATE(reservation_date), DATE(creation_timestamp), DAY) <= 14 
               THEN covers_created_on_day ELSE 0 END) as avg_covers_14_days_out,
      AVG(CASE WHEN DATE_DIFF(DATE(reservation_date), DATE(creation_timestamp), DAY) <= 30 
               THEN covers_created_on_day ELSE 0 END) as avg_covers_30_days_out,
      AVG(CASE WHEN DATE_DIFF(DATE(reservation_date), DATE(creation_timestamp), DAY) <= 60 
               THEN covers_created_on_day ELSE 0 END) as avg_covers_60_days_out
    FROM ReservationsByCreationDate
    GROUP BY 1, 2, 3
  )

-- Final Step: Join the scaffold with the daily created reservations and calculate enhanced features
SELECT
  t.shift_date,
  t.venue_key_id,
  t.persistent_id,
  t.snapshot_timestamp,
  t.shift_category,

  -- === CORE MODEL FEATURES ===
  -- Time-based features
  t.day_prior,
  EXTRACT(DAYOFWEEK FROM t.shift_date) AS day_of_week,
  EXTRACT(MONTH FROM t.shift_date) AS month,
  EXTRACT(YEAR FROM t.shift_date) AS year,
  
  -- Capacity features
  t.util_capacity,
  
  -- Current booking features (cumulative up to snapshot date)
  SUM(COALESCE(r.covers_created_on_day, 0)) OVER (
    PARTITION BY t.venue_key_id, t.persistent_id, t.shift_date
    ORDER BY t.snapshot_timestamp ASC
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS covers_as_of_date,
  
  -- Enhanced booking velocity features
  SUM(COALESCE(r.reservations_created_on_day, 0)) OVER (
    PARTITION BY t.venue_key_id, t.persistent_id, t.shift_date
    ORDER BY t.snapshot_timestamp ASC
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS reservations_as_of_date,
  
  -- Average party size (as of snapshot date)
  CASE 
    WHEN SUM(COALESCE(r.reservations_created_on_day, 0)) OVER (
           PARTITION BY t.venue_key_id, t.persistent_id, t.shift_date
           ORDER BY t.snapshot_timestamp ASC
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
         ) > 0
    THEN SUM(COALESCE(r.covers_created_on_day, 0)) OVER (
           PARTITION BY t.venue_key_id, t.persistent_id, t.shift_date
           ORDER BY t.snapshot_timestamp ASC
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
         ) / SUM(COALESCE(r.reservations_created_on_day, 0)) OVER (
               PARTITION BY t.venue_key_id, t.persistent_id, t.shift_date
               ORDER BY t.snapshot_timestamp ASC
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
             )
    ELSE 0
  END AS avg_party_size_as_of_date,
  
  -- Capacity utilization as of snapshot date
  CASE 
    WHEN t.util_capacity > 0 
    THEN SUM(COALESCE(r.covers_created_on_day, 0)) OVER (
           PARTITION BY t.venue_key_id, t.persistent_id, t.shift_date
           ORDER BY t.snapshot_timestamp ASC
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
         ) / t.util_capacity 
    ELSE 0 
  END AS utilization_as_of_date,
  
  -- Recent booking velocity (covers in last 7 days before snapshot)
  SUM(CASE WHEN DATE_DIFF(t.snapshot_timestamp, r.creation_timestamp, DAY) <= 7 
           THEN COALESCE(r.covers_created_on_day, 0) ELSE 0 END) OVER (
    PARTITION BY t.venue_key_id, t.persistent_id, t.shift_date
    ORDER BY t.snapshot_timestamp ASC
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS covers_last_7_days,
  
  -- Historical pattern features
  COALESCE(h.avg_covers_7_days_out, 0) AS historical_avg_covers_7d,
  COALESCE(h.avg_covers_14_days_out, 0) AS historical_avg_covers_14d,
  COALESCE(h.avg_covers_30_days_out, 0) AS historical_avg_covers_30d,
  COALESCE(h.avg_covers_60_days_out, 0) AS historical_avg_covers_60d,
  
  -- Seasonal and day-of-week indicators
  CASE WHEN EXTRACT(MONTH FROM t.shift_date) IN (12, 1, 2) THEN 1 ELSE 0 END AS is_winter,
  CASE WHEN EXTRACT(MONTH FROM t.shift_date) IN (3, 4, 5) THEN 1 ELSE 0 END AS is_spring,
  CASE WHEN EXTRACT(MONTH FROM t.shift_date) IN (6, 7, 8) THEN 1 ELSE 0 END AS is_summer,
  CASE WHEN EXTRACT(MONTH FROM t.shift_date) IN (9, 10, 11) THEN 1 ELSE 0 END AS is_fall,
  CASE WHEN EXTRACT(DAYOFWEEK FROM t.shift_date) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend,
  
  -- Time horizon indicators
  CASE WHEN t.day_prior <= 7 THEN 1 ELSE 0 END AS is_within_week,
  CASE WHEN t.day_prior <= 30 THEN 1 ELSE 0 END AS is_within_month,
  CASE WHEN t.day_prior <= 90 THEN 1 ELSE 0 END AS is_within_quarter,

  -- === TARGET VARIABLE ===
  t.is_hot

FROM TrainingScaffold AS t
LEFT JOIN ReservationsByCreationDate AS r
  ON t.venue_key_id = r.venue_key_id
  AND t.persistent_id = r.shift_persistent_id
  AND t.shift_date = r.reservation_date
  AND t.snapshot_timestamp >= r.creation_timestamp -- All reservations created ON or BEFORE the snapshot date
LEFT JOIN HistoricalBookingVelocity AS h
  ON EXTRACT(DAYOFWEEK FROM t.shift_date) = h.day_of_week
  AND EXTRACT(MONTH FROM t.shift_date) = h.month
  AND t.persistent_id = h.shift_persistent_id
WHERE 
  -- Quality filters
  t.snapshot_timestamp <= t.shift_date  -- Snapshot must be before or on shift date
  AND t.snapshot_timestamp >= DATE_SUB(t.shift_date, INTERVAL 365 DAY)  -- Don't look back more than a year
ORDER BY
  t.shift_date, t.persistent_id, t.snapshot_timestamp;