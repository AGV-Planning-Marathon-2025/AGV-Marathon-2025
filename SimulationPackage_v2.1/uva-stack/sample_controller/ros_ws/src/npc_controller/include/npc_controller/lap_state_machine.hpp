#ifndef NPC_CONTROLLER__LAP_STATE_MACHINE_HPP_
#define NPC_CONTROLLER__LAP_STATE_MACHINE_HPP_

#include <chrono>
#include <iostream>

#include "npc_controller/rc_states.hpp"

using namespace std::chrono;

namespace controller{

struct LapStateSpeeds{
  double yellow_flag_box_exit;
  double yellow_flag_pit_row;
  double yellow_flag_pit_lane;
  double yellow_flag_on_track;
  double green_flag;
  double black_flag;
};

struct LapStateLocs{
  double pit_transition_out_loc;
  double pit_transition_in_loc;
  double pit_entry_dec_loc;
  double pit_inc_speed_loc;
  double pit_dec_speed_loc;
  double pit_stop;
};

struct LapStateOutputs{
  double des_vel;
  std::string path;
  bool attacker;
  bool overtake;
};

struct LapStateInputs{
  CTState ct_state;
  VehicleFlag vehicle_flag;
  TrackFlag track_flag;
  TrackLocation track_loc;
  LapState lap_state;
  double target_speed;
  double current_speed;
  double pit_lane_s;
  double center_line_s;
  LapStateSpeeds speeds;
  LapStateLocs locs;
};

class LapStateMachine{
  public:
      LapStateMachine() = default;
      ~LapStateMachine() = default;

    void transition(LapStateInputs &inputs, LapStateOutputs &outputs)
    {

      if (inputs.ct_state == CTState::CT10_COORD_STOP) {
        outputs.des_vel = 0.0;
        prev_ct_state_ = inputs.ct_state;
        previous_center_s_ = inputs.center_line_s;
        previous_pit_s_ = inputs.pit_lane_s;
        return;
      }

      std_msgs::msg::Bool planner_flag;
      if (inputs.track_flag == TrackFlag::TF8_WAVING_GREEN) {
        outputs.overtake = true;
      } else {
        outputs.overtake = false;
      }

      if (inputs.vehicle_flag == VehicleFlag::VF7_ATTACKER) {
        outputs.attacker = true;
        outputs.des_vel = inputs.target_speed;
      } else if (inputs.vehicle_flag == VehicleFlag::VF5_DEFENDER) {
        outputs.attacker = false;
        outputs.des_vel = inputs.target_speed;
      }

      // Black flag states 
      if (inputs.vehicle_flag == VehicleFlag::VF2_BLACK && inputs.lap_state == LapState::LS3_ON_RACELINE) { // Black Flag Received
        outputs.des_vel = inputs.speeds.black_flag;
        inputs.lap_state = LapState::LS4_BLACK_SLOWDOWN;
      } else if (inputs.lap_state == LapState::LS4_BLACK_SLOWDOWN && inputs.vehicle_flag != VehicleFlag::VF2_BLACK) { // Transition out of Black
        if (inputs.track_flag == TrackFlag::TF3_YELLOW) {
          outputs.des_vel = inputs.speeds.yellow_flag_on_track;
        } else if (inputs.track_flag == TrackFlag::TF4_GREEN) {
          outputs.des_vel = inputs.speeds.green_flag;
        }
        inputs.lap_state = LapState::LS3_ON_RACELINE;
      } else if (inputs.lap_state == LapState::LS4_BLACK_SLOWDOWN && inputs.vehicle_flag == VehicleFlag::VF2_BLACK) { // Black Flag slow to pit speed
        if (center_passed_point(inputs.center_line_s, inputs.locs.pit_entry_dec_loc)) {
          outputs.des_vel = inputs.speeds.yellow_flag_pit_lane;
          inputs.lap_state = LapState::LS5_BLACK_SWITCH_TO_PIT;
        }
      } else if (inputs.lap_state == LapState::LS5_BLACK_SWITCH_TO_PIT 
                  && inputs.current_speed <= inputs.speeds.yellow_flag_pit_lane +5.0
                  && center_passed_point(inputs.center_line_s, inputs.locs.pit_transition_in_loc)) {
        outputs.path = "pits";
        inputs.lap_state = LapState::LS6_PIT_ENTRY;
      //} else if (lap_state == LapState::LS6_PIT_ENTRY && dist <= 20.0) {
      } else if (inputs.lap_state == LapState::LS6_PIT_ENTRY 
                  && pit_passed_point(inputs.pit_lane_s, inputs.locs.pit_stop-30.0)) {
        // TODO: Determine a better way of telling when the car has returned to the box
        outputs.des_vel = 0.0;
        inputs.lap_state = LapState::LS2_PIT_EXIT;
      }

      // TODO: Allow LapStateMachine to be restarted while on track, ignoring initial states
      // Pit exit states
      else if (inputs.ct_state == CTState::CT8_CAUTION && prev_ct_state_ != CTState::CT8_CAUTION && inputs.lap_state == LapState::LS0_IN_BOX) { // Initial Yellow Flag
        outputs.path = "pits";
        outputs.des_vel = inputs.speeds.yellow_flag_box_exit;
        inputs.lap_state = LapState::LS1_LEAVE_BOX;
      } else if (inputs.lap_state == LapState::LS1_LEAVE_BOX) { // Vehicle has moved from box to pit line
        outputs.des_vel = inputs.speeds.yellow_flag_pit_row;
        inputs.lap_state = LapState::LS2_PIT_EXIT;
      } else if (inputs.lap_state == LapState::LS2_PIT_EXIT
                && pit_passed_point(inputs.pit_lane_s, inputs.locs.pit_transition_in_loc)) {
        // Vehicle has reached the defined polygon for the first time
        outputs.path = "race";
        inputs.lap_state = LapState::LS3_ON_RACELINE;
        if (inputs.ct_state == CTState::CT9_NOM_RACE) {
          outputs.des_vel = inputs.speeds.green_flag;
        } else if (inputs.ct_state == CTState::CT8_CAUTION) {
          outputs.des_vel = inputs.speeds.yellow_flag_on_track;
        }
      }

      // On-track flag states
      else if (inputs.ct_state == CTState::CT8_CAUTION && prev_ct_state_ != CTState::CT8_CAUTION) { // Yellow Flag
        if (inputs.lap_state == LapState::LS2_PIT_EXIT) { // Exiting Pits
          if (inputs.track_loc == TrackLocation::LOC_PIT_SPEEDUP) { // Out of slow zone
            outputs.des_vel = inputs.speeds.yellow_flag_pit_lane;
            
          } else { // Inside slow zone
            outputs.des_vel = inputs.speeds.yellow_flag_pit_row; // Yellow past boxes
            
          }
        } else if (inputs.lap_state == LapState::LS3_ON_RACELINE) { // On Raceline
          outputs.des_vel = inputs.speeds.yellow_flag_on_track;
          
        } else if (inputs.lap_state == LapState::LS4_BLACK_SLOWDOWN) {
          outputs.des_vel = inputs.speeds.black_flag;
          
        } else if (inputs.lap_state == LapState::LS5_BLACK_SWITCH_TO_PIT) {
          outputs.des_vel = inputs.speeds.yellow_flag_pit_lane;
          
        } else if (inputs.lap_state == LapState::LS6_PIT_ENTRY) {
          if (inputs.track_loc == TrackLocation::LOC_PIT_SLOWDOWN) { // Inside slow zone
            outputs.des_vel = inputs.speeds.yellow_flag_pit_row; // Yellow past boxes
            
          } else { // Out of slow zone
            outputs.des_vel = inputs.speeds.yellow_flag_pit_lane;
            
          }
        }
      } else if (inputs.ct_state == CTState::CT9_NOM_RACE && prev_ct_state_ != CTState::CT9_NOM_RACE) { // Green Flag
        if (inputs.lap_state == LapState::LS2_PIT_EXIT) {
          outputs.des_vel = inputs.speeds.yellow_flag_pit_lane;
          
        } else if (inputs.lap_state == LapState::LS3_ON_RACELINE) { // In Pits or On Raceline
          outputs.des_vel = inputs.speeds.green_flag;
          
        } else if (inputs.lap_state == LapState::LS4_BLACK_SLOWDOWN) {
          outputs.des_vel = inputs.speeds.black_flag;
          
        } else if (inputs.lap_state == LapState::LS5_BLACK_SWITCH_TO_PIT || inputs.lap_state == LapState::LS6_PIT_ENTRY) {
          outputs.des_vel = inputs.speeds.yellow_flag_pit_lane;
          
        }
      }

      // Track Location
      if ((pit_passed_point(inputs.pit_lane_s, inputs.locs.pit_transition_in_loc) 
          || center_passed_point(inputs.center_line_s, inputs.locs.pit_transition_out_loc))
          && inputs.track_loc != TrackLocation::LOC_PIT_ENTRY_EXIT) { // pit exit/entry
        inputs.track_loc = TrackLocation::LOC_PIT_ENTRY_EXIT;
      } else if (pit_passed_point(inputs.pit_lane_s, inputs.locs.pit_inc_speed_loc) && inputs.track_loc != TrackLocation::LOC_PIT_SPEEDUP) { // Past pitboxes
        inputs.track_loc = TrackLocation::LOC_PIT_SPEEDUP;
        if (inputs.lap_state == LapState::LS2_PIT_EXIT) {
          outputs.des_vel = inputs.speeds.yellow_flag_pit_lane;
          
        }
      } else if (pit_passed_point(inputs.pit_lane_s, inputs.locs.pit_dec_speed_loc) && inputs.track_loc != TrackLocation::LOC_PIT_SLOWDOWN) { // Approaching pitboxes
        inputs.track_loc = TrackLocation::LOC_PIT_SLOWDOWN;
        if (inputs.lap_state == LapState::LS6_PIT_ENTRY) {
          outputs.des_vel = inputs.speeds.yellow_flag_pit_row; // Yellow past boxes
          
        }
      }

      prev_ct_state_ = inputs.ct_state;
      previous_center_s_ = inputs.center_line_s;
      previous_pit_s_ = inputs.pit_lane_s;
  }

  bool pit_passed_point(double s, double s_check) {
    if(s >= s_check && previous_pit_s_ <= s_check){
      return true;
    } else if(s >= s_check && previous_pit_s_ > s){
      return true;
    } else {
      return false;
    }
  }

  bool center_passed_point(double s, double s_check){
    if(s >= s_check && previous_center_s_ <= s_check){
      return true;
    } else if(s >= s_check && previous_center_s_ > s){
      return true;
    } else {
      return false;
    }
  }


  double previous_pit_s_;
  double previous_center_s_;
  CTState prev_ct_state_;
  
};


} // namespace controller

#endif // NPC_CONTROLLER__LAP_STATE_MACHINE_HPP_