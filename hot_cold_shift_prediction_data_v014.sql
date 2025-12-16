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
          UNNEST(GENERATE_DATE_ARRAY(DATE_ADD(CURRENT_DATE(), INTERVAL -12 YEAR), DATE_ADD(CURRENT_DATE(), INTERVAL 365 DAY))) AS DATE
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

  -- Calculate capacity for all relevant future shifts (enhanced date range)
  FutureShiftCapacity AS (
    SELECT *
    FROM ShiftCapacityPerDay
    WHERE DATE(DATE) BETWEEN CURRENT_DATE() AND DATE_ADD(CURRENT_DATE(), INTERVAL 365 DAY)
      AND venue_key_id = 14301284
  ),

  -- Aggregate all current bookings for those future shifts (enhanced with booking timing)
  CurrentBookings AS (
    SELECT
      venue_key_id,
      shift_persistent_id,
      date AS reservation_date,
      SUM(max_guests) as current_total_covers,
      -- Additional metrics for enhanced prediction
      COUNT(*) as total_reservations,
      AVG(max_guests) as avg_party_size,
      MIN(created) as first_booking_timestamp,
      MAX(created) as latest_booking_timestamp
    FROM `sevenrooms-datawarehouse.sevenrooms_datawarehouse_prod.nightloop_ReservationActual_Latest`
    WHERE status NOT IN ('CANCELED')
      AND venue_key_id = 14301284
      AND created IS NOT NULL
      AND created <= CURRENT_TIMESTAMP()
    GROUP BY 1, 2, 3
  ),

  -- Enhanced historical booking patterns for better prediction
  HistoricalBookingPatterns AS (
    SELECT
      EXTRACT(DAYOFWEEK FROM date) as day_of_week,
      EXTRACT(MONTH FROM date) as month,
      shift_persistent_id,
      -- Average booking velocity in the last 30, 60, 90 days before shift
      AVG(CASE WHEN DATE_DIFF(date, DATE(created), DAY) <= 30 THEN max_guests ELSE 0 END) as avg_covers_last_30_days,
      AVG(CASE WHEN DATE_DIFF(date, DATE(created), DAY) <= 60 THEN max_guests ELSE 0 END) as avg_covers_last_60_days,
      AVG(CASE WHEN DATE_DIFF(date, DATE(created), DAY) <= 90 THEN max_guests ELSE 0 END) as avg_covers_last_90_days,
      -- Booking lead times
      AVG(DATE_DIFF(date, DATE(created), DAY)) as avg_lead_time_days
    FROM `sevenrooms-datawarehouse.sevenrooms_datawarehouse_prod.nightloop_ReservationActual_Latest`
    WHERE status NOT IN ('CANCELED')
      AND venue_key_id = 14301284
      AND created IS NOT NULL
      AND date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)  -- Look back 2 years for patterns
      AND date < CURRENT_DATE()  -- Only historical data
    GROUP BY 1, 2, 3
  )

-- Final Step: Generate the enhanced feature set for prediction
SELECT
  s.DATE AS shift_date,
  s.venue_key_id,
  s.persistent_id,
  s.shift_category,
  
  -- === CORE MODEL FEATURES ===
  -- Time-based features
  DATE_DIFF(DATE(s.DATE), CURRENT_DATE(), DAY) AS day_prior,
  EXTRACT(DAYOFWEEK FROM s.DATE) AS day_of_week,
  EXTRACT(MONTH FROM s.DATE) AS month,
  EXTRACT(YEAR FROM s.DATE) AS year,
  
  -- Capacity features
  s.util_capacity,
  
  -- Current booking features
  COALESCE(r.current_total_covers, 0) AS covers_as_of_date,
  COALESCE(r.total_reservations, 0) AS current_reservation_count,
  COALESCE(r.avg_party_size, 0) AS current_avg_party_size,
  
  -- Capacity utilization
  CASE 
    WHEN s.util_capacity > 0 THEN COALESCE(r.current_total_covers, 0) / s.util_capacity 
    ELSE 0 
  END AS current_utilization_rate,
  
  -- Booking timing features
  CASE 
    WHEN r.first_booking_timestamp IS NOT NULL 
    THEN DATE_DIFF(DATE(s.DATE), DATE(r.first_booking_timestamp), DAY)
    ELSE NULL 
  END AS days_since_first_booking,
  
  CASE 
    WHEN r.latest_booking_timestamp IS NOT NULL 
    THEN DATE_DIFF(CURRENT_DATE(), DATE(r.latest_booking_timestamp), DAY)
    ELSE NULL 
  END AS days_since_latest_booking,
  
  -- Historical pattern features
  COALESCE(h.avg_covers_last_30_days, 0) AS historical_avg_covers_30d,
  COALESCE(h.avg_covers_last_60_days, 0) AS historical_avg_covers_60d,
  COALESCE(h.avg_covers_last_90_days, 0) AS historical_avg_covers_90d,
  COALESCE(h.avg_lead_time_days, 0) AS historical_avg_lead_time,
  
  -- Seasonal indicators
  CASE 
    WHEN EXTRACT(MONTH FROM s.DATE) IN (12, 1, 2) THEN 1 ELSE 0 
  END AS is_winter,
  
  CASE 
    WHEN EXTRACT(MONTH FROM s.DATE) IN (3, 4, 5) THEN 1 ELSE 0 
  END AS is_spring,
  
  CASE 
    WHEN EXTRACT(MONTH FROM s.DATE) IN (6, 7, 8) THEN 1 ELSE 0 
  END AS is_summer,
  
  CASE 
    WHEN EXTRACT(MONTH FROM s.DATE) IN (9, 10, 11) THEN 1 ELSE 0 
  END AS is_fall,
  
  -- Weekend indicator
  CASE 
    WHEN EXTRACT(DAYOFWEEK FROM s.DATE) IN (1, 7) THEN 1 ELSE 0 
  END AS is_weekend

FROM FutureShiftCapacity AS s
LEFT JOIN CurrentBookings AS r
  ON s.venue_key_id = r.venue_key_id
  AND s.persistent_id = r.shift_persistent_id
  AND s.DATE = r.reservation_date
LEFT JOIN HistoricalBookingPatterns AS h
  ON EXTRACT(DAYOFWEEK FROM s.DATE) = h.day_of_week
  AND EXTRACT(MONTH FROM s.DATE) = h.month
  AND s.persistent_id = h.shift_persistent_id
WHERE
  s.util_capacity IS NOT NULL 
  AND s.util_capacity > 0
ORDER BY
  s.DATE;