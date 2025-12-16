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
          var lastTimeRange = keys[keys.length - 1];
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

-- Main query combining capacity calculation with actual reservation data
WITH
  -- CTE #1: Your powerful query to calculate the theoretical capacity of each shift per day.
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
          shifts.category
        FROM
          date_series
        LEFT JOIN `sevenrooms-datawarehouse.sevenrooms_datawarehouse_prod.sr_ShiftDefault_Latest` shifts ON 1 = 1
        LEFT JOIN shifts_excluded_dates ON shifts.venue_key_id = shifts_excluded_dates.venue_id AND shifts.int_Id = shifts_excluded_dates.int_id AND date_series.DATE = shifts_excluded_dates.excluded_date
        LEFT JOIN shift_days_of_week ON shifts.venue_key_id = shift_days_of_week.venue_id AND shifts.int_id = shift_days_of_week.int_id
        WHERE
          shifts.venue_key_id = 14301284 -- OPTIMIZATION: Filter early
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
          shifts.category
        FROM
          date_series
        LEFT JOIN `sevenrooms-datawarehouse.sevenrooms_datawarehouse_prod.sr_ShiftOverride_Latest` shifts ON shifts.DATE = date_series.DATE
        WHERE
          shifts.venue_key_id = 14301284 -- OPTIMIZATION: Filter early
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
          shift_by_date.category,
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
          CASE
            WHEN end_time < normalized_start_time THEN TIMESTAMP_ADD(end_time, INTERVAL 24 HOUR)
            ELSE end_time
          END AS normalized_end_time,
          shift_util_temp.*
        FROM (
          SELECT
            shifts.DATE,
            shifts.int_id,
            shifts.name,
            shifts.persistent_id,
            shifts.category,
            shifts.deleted,
            shifts.effective_start_date,
            shifts.effective_end_date,
            TIMESTAMP('1970-01-01 00:00:00') + (start_time - TIMESTAMP('1900-01-01 00:00:00')) AS normalized_start_time,
            TIMESTAMP('1970-01-01 00:00:00') + (end_time - TIMESTAMP('1900-01-01 00:00:00')) AS end_time,
            shifts.venue_key_id,
            shifts.interval_minutes,
            shifts.day_of_week,
            duration_minutes_by_party_size_str,
            duration_dict_ranges_str,
            GetShiftCapacity(custom_pacing_str, covers_per_seating_interval, shift_start_time_min, shift_end_time_min, interval_minutes) AS max_shift_capacity,
            SUM(GetShiftwiseTableCapacity(duration_dict_ranges_str, table_size, shift_start_time_min, shift_end_time_min)) AS calculated_capacity
          FROM
            shift_denorm_by_date shifts
          GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ) shift_util_temp
      )
    SELECT
      *,
      EXTRACT(DAY FROM normalized_end_time) AS normalized_end_time_dom,
      EXTRACT(HOUR FROM normalized_end_time) * 60 + EXTRACT(MINUTE FROM normalized_end_time) AS normalized_end_time_minutes,
      EXTRACT(HOUR FROM normalized_start_time) * 60 + EXTRACT(MINUTE FROM normalized_start_time) AS normalized_start_time_minutes,
      DATE_TRUNC(DATE, WEEK) AS date_partition
    FROM
      shift_util
),

-- CTE #2: Aggregate actual reservation data to get booking pace and final cover count.
ReservationsAggregated AS (
  SELECT
    r.venue_key_id,
    r.shift_persistent_id,
    TIMESTAMP_TRUNC(r.marked_time, DAY) AS reservation_date,
    SUM(r.max_guests) AS final_actual_covers,

    -- Calculate cumulative covers booked by various points in time before the reservation date
    SUM(IF(DATE_DIFF(DATE(r.marked_time), DATE(r.created), DAY) >= 30, r.max_guests, 0)) AS covers_booked_30_days_prior,
    SUM(IF(DATE_DIFF(DATE(r.marked_time), DATE(r.created), DAY) >= 14, r.max_guests, 0)) AS covers_booked_14_days_prior,
    SUM(IF(DATE_DIFF(DATE(r.marked_time), DATE(r.created), DAY) >= 7, r.max_guests, 0)) AS covers_booked_7_days_prior,
    SUM(IF(DATE_DIFF(DATE(r.marked_time), DATE(r.created), DAY) >= 3, r.max_guests, 0)) AS covers_booked_3_days_prior,
    SUM(IF(DATE_DIFF(DATE(r.marked_time), DATE(r.created), DAY) >= 1, r.max_guests, 0)) AS covers_booked_1_day_prior
  FROM
    `sevenrooms-datawarehouse.sevenrooms_datawarehouse_prod.nightloop_ReservationActual_Latest` AS r
  WHERE
    r.venue_key_id = 14301284 -- OPTIMIZATION & CORRECTION: Filter by the correct venue ID field
    AND r.status NOT IN ('CANCELED')
    AND r.marked_time IS NOT NULL
    AND r.created IS NOT NULL
  GROUP BY
    1, 2, 3
)

-- Final Step: Join capacity with actuals and define the target variable 'is_hot'.
SELECT
  s.DATE,
  s.venue_key_id,
  v.name AS venue_name,
  s.name AS shift_name,
  s.persistent_id AS shift_persistent_id,
  s.util_capacity,
  COALESCE(r.final_actual_covers, 0) AS final_actual_covers,
  COALESCE(r.covers_booked_30_days_prior, 0) AS covers_booked_30_days_prior,
  COALESCE(r.covers_booked_14_days_prior, 0) AS covers_booked_14_days_prior,
  COALESCE(r.covers_booked_7_days_prior, 0) AS covers_booked_7_days_prior,
  COALESCE(r.covers_booked_3_days_prior, 0) AS covers_booked_3_days_prior,
  COALESCE(r.covers_booked_1_day_prior, 0) AS covers_booked_1_day_prior,

  -- Define the target variable: 1 for hot, 0 for cold.
  -- A shift is "hot" if its final covers exceeded 85% of its utilization capacity.
  CASE
    WHEN s.util_capacity > 0 AND (COALESCE(r.final_actual_covers, 0) / s.util_capacity) > 0.6 THEN 1
    ELSE 0
  END AS is_hot

FROM
  ShiftCapacityPerDay AS s
LEFT JOIN
  ReservationsAggregated AS r ON s.venue_key_id = r.venue_key_id
  AND s.persistent_id = r.shift_persistent_id
  AND s.DATE = r.reservation_date
LEFT JOIN
  `sevenrooms-datawarehouse.sevenrooms_datawarehouse_prod.nightloop_Venue` AS v ON s.venue_key_id = v.__key__.id
WHERE
  -- Filter capacity results to the specific venue as well
  s.venue_key_id = 14301284
  -- Use only historical data for training the model
  AND DATE(s.DATE) < CURRENT_DATE()
  AND s.util_capacity IS NOT NULL -- Ensure we only have shifts with a calculated capacity
ORDER BY
  s.DATE DESC, s.venue_key_id;

