import logging
import random
import sys
import numpy as np
from psychopy import visual, core, event

from modules.config import *

def run_neurofeedback_experiment(neurofeedback_processor):
    """
    Run the neurofeedback experiment with visualizations using PsychoPy.
    
    Parameters:
    -----------
    neurofeedback_processor : NeurofeedbackProcessor
        The processor instance that handles EEG signal processing
    """
    from pylsl import StreamInfo, StreamOutlet
    # Create LSL Marker stream
    try:
        marker_info = StreamInfo(MARKER_STREAM_NAME, "Markers", 1, 0, "string", "marker_id")
        marker_outlet = StreamOutlet(marker_info)
        logging.info("Marker stream created.")
    except Exception as e:
        logging.exception("Failed to create LSL Marker stream:")
        sys.exit(1)

    # PsychoPy setup
    try:
        from psychopy import visual, core, event, clock
        
        # Set up monitor for optimal performance
        win = visual.Window(
            size=(1024, 768),
            color="black",
            units="norm",
            fullscr=False,
            allowGUI=False,  # Disable GUI for better performance
            waitBlanking=False  # Disable waiting for screen refresh for lower latency
        )
        win.recordFrameIntervals = True  # Track frame timing
        
        logging.info("PsychoPy window created.")
    except Exception as e:
        logging.exception("Error setting up PsychoPy window:")
        sys.exit(1)

    # Create visual stimuli
    instruction_text = visual.TextStim(win, text="", height=0.15, color="white")
    ready_text = visual.TextStim(win, text="Get Ready", height=0.15, color="white")
    countdown_text = visual.TextStim(win, text="", height=0.2, color="white", pos=(0, 0.2))
    
    # Create channel indicator
    channel_text = visual.TextStim(win, text="", height=0.05, color="white",
                               pos=(0, -0.7))
    
    # Create Mu band feedback meter (right side) - similar to train_response.py
    meter_width = 0.2
    meter_height = 0.6
    mu_meter_pos = (0.7, 0)  # Position on right side of screen
    mu_meter_bg = visual.Rect(
        win=win, 
        width=meter_width, 
        height=meter_height, 
        pos=mu_meter_pos, 
        fillColor='gray', 
        lineColor='white',
        autoLog=False
    )
    mu_meter_fill = visual.Rect(
        win=win, 
        width=meter_width, 
        height=0, 
        pos=mu_meter_pos, 
        fillColor='green',
        autoLog=False
    )
    mu_meter_text = visual.TextStim(
        win=win, 
        text="Mu Band Power", 
        height=0.05,
        pos=(mu_meter_pos[0], mu_meter_pos[1] + meter_height/2 + 0.1),
        autoLog=False
    )
    
    # Create Beta band feedback meter (left side)
    beta_meter_pos = (-0.7, 0)  # Position on left side of screen
    beta_meter_bg = visual.Rect(
        win=win, 
        width=meter_width, 
        height=meter_height, 
        pos=beta_meter_pos, 
        fillColor='gray', 
        lineColor='white',
        autoLog=False
    )
    beta_meter_fill = visual.Rect(
        win=win, 
        width=meter_width, 
        height=0, 
        pos=beta_meter_pos, 
        fillColor='blue',
        autoLog=False
    )
    beta_meter_text = visual.TextStim(
        win=win, 
        text="Beta Band Power", 
        height=0.05,
        pos=(beta_meter_pos[0], beta_meter_pos[1] + meter_height/2 + 0.1),
        autoLog=False
    )
    
    # Create central feedback bar with optimized settings (keep this from original)
    feedback_bar_bg = visual.Rect(
        win=win,
        width=FEEDBACK_BAR_MAX_WIDTH,
        height=0.2,
        fillColor="gray",
        lineColor="white",
        pos=(0, -0.5),
        autoLog=False
    )
    
    feedback_bar = visual.Rect(
        win=win,
        width=0,  # Will be updated based on ERD
        height=0.15,
        fillColor="green",
        lineColor=None,
        pos=(-FEEDBACK_BAR_MAX_WIDTH/2, -0.5),  # Centered position
        anchor="left",  # Anchor to left for width updates
        autoLog=False
    )

    # Generate randomized trial list
    trials = ["right", "left"] * (NUM_TRIALS // 2)
    random.shuffle(trials)
    
    # Show initial instructions
    instruction_text.text = "Motor Imagery Neurofeedback\n\nImagine moving your hand when instructed\n\nThe feedback bars will show your brain activity\n\nPress SPACE to begin"
    instruction_text.draw()
    win.flip()
    event.waitKeys(keyList=["space"])
    
    # Initial 10-second baseline collection
    instruction_text.text = "Collecting baseline\n\nPlease relax and remain still"
    neurofeedback_processor.start_initial_baseline_collection()
    
    baseline_clock = core.Clock()
    baseline_clock.reset()
    
    # Countdown for initial baseline
    while baseline_clock.getTime() < INITIAL_BASELINE_DURATION:
        remaining = int(INITIAL_BASELINE_DURATION - baseline_clock.getTime())
        instruction_text.draw()
        countdown_text.text = f"{remaining}s"
        countdown_text.draw()
        win.flip()
        
        # Check for quit
        if event.getKeys(keyList=["escape"]):
            win.close()
            neurofeedback_processor.stop_initial_baseline_collection()
            return
    
    neurofeedback_processor.stop_initial_baseline_collection()
    
    # Indicate baseline completion
    instruction_text.text = "Baseline Completed\n\nThe experiment will begin shortly"
    instruction_text.draw()
    win.flip()
    core.wait(2.0)

    # Main experiment loop
    for trial_num, hand in enumerate(trials, 1):
        # Set the active hand for this trial
        neurofeedback_processor.set_active_hand(hand)
        
        # Update the channel indicator
        channel_used = "CH3" if hand == "right" else "CH6"
        channel_text.text = f"Using {channel_used} (Contralateral Hemisphere)"
        
        # Display get ready message
        ready_text.draw()
        channel_text.draw()
        win.flip()
        core.wait(1.0)
        
        # Show instruction (right or left hand)
        instruction_text.text = f"{hand.upper()} HAND"
        instruction_text.draw()
        channel_text.draw()
        win.flip()
        core.wait(INSTRUCTION_DURATION)
        
        # Start collecting baseline data 3 seconds before the START cue
        # We use a separate clock to time the baseline collection
        baseline_clock = core.Clock()
        baseline_clock.reset()
        neurofeedback_processor.start_baseline_collection()
        
        # Wait for baseline collection (-3s to -1s before START)
        # Continue showing instruction during this time
        while baseline_clock.getTime() < abs(BASELINE_START - BASELINE_END):  # 2 seconds
            instruction_text.draw()
            channel_text.draw()
            win.flip()
            
            # Check for quit
            if event.getKeys(keyList=["escape"]):
                win.close()
                return
        
        # Stop baseline collection and prepare for processing
        neurofeedback_processor.stop_baseline_collection()
        
        # Show START cue and send marker
        instruction_text.text = "START"
        instruction_text.draw()
        channel_text.draw()
        # Schedule marker to be sent on next flip
        marker_val = MARKER_RIGHT if hand == "right" else MARKER_LEFT
        win.callOnFlip(lambda m=marker_val: marker_outlet.push_sample([m]))
        win.flip()
        
        # Start real-time processing after the START cue
        neurofeedback_processor.start_processing()
        
        # Initialize feedback timer and display timer
        feedback_timer = core.Clock()
        display_timer = core.Clock()
        feedback_timer.reset()
        display_timer.reset()
        
        # Continuous feedback during imagery period
        while feedback_timer.getTime() < IMAGERY_DURATION:
            # Only update display at the target refresh rate
            if display_timer.getTime() >= DISPLAY_UPDATE_INTERVAL:
                # Reset display timer
                display_timer.reset()
                
                # Get current ERD values with smoothing for visual appeal
                smoothed_mu = neurofeedback_processor.get_smoothed_erd_mu()
                smoothed_beta = neurofeedback_processor.get_smoothed_erd_beta()
                
                # Update central feedback bar width based on mu ERD value
                bar_width = min(smoothed_mu * FEEDBACK_BAR_MAX_WIDTH, FEEDBACK_BAR_MAX_WIDTH)
                feedback_bar.width = bar_width
                
                # Update mu meter height
                mu_meter_fill.height = smoothed_mu * meter_height
                mu_meter_fill.pos = (mu_meter_pos[0], mu_meter_pos[1] - (meter_height/2) + (smoothed_mu * meter_height/2))
                
                # Update beta meter height
                beta_meter_fill.height = smoothed_beta * meter_height
                beta_meter_fill.pos = (beta_meter_pos[0], beta_meter_pos[1] - (meter_height/2) + (smoothed_beta * meter_height/2))
                
                # Update central bar color based on mu ERD value
                if smoothed_mu < 0.2:
                    feedback_bar.fillColor = "red"
                elif smoothed_mu < 0.6:
                    feedback_bar.fillColor = "yellow"
                else:
                    feedback_bar.fillColor = "green"
                
                # Draw feedback
                instruction_text.text = "HOLD"
                instruction_text.draw()
                
                # Draw mu and beta meters
                mu_meter_bg.draw()
                mu_meter_fill.draw()
                mu_meter_text.draw()
                beta_meter_bg.draw()
                beta_meter_fill.draw()
                beta_meter_text.draw()
                
                # Draw central feedback bar
                feedback_bar_bg.draw()
                feedback_bar.draw()
                
                # Draw channel indicator
                channel_text.draw()
                
                win.flip()
            
            # Process any events to prevent blocking
            if event.getKeys(keyList=["escape"]):
                win.close()
                return
        
        # Stop real-time processing
        neurofeedback_processor.stop_processing()
        
        # Show STOP and send stop marker
        instruction_text.text = "STOP"
        instruction_text.draw()
        channel_text.draw()
        win.callOnFlip(lambda: marker_outlet.push_sample([MARKER_STOP]))
        win.flip()
        core.wait(1.0)
        
        # Inter-trial interval
        win.flip()  # clear screen
        core.wait(INTER_TRIAL_INTERVAL)
        
        # Report frame timing stats
        if win.recordFrameIntervals:
            frame_times = win.frameIntervals
            if frame_times:
                mean_frame_time = np.mean(frame_times)
                std_frame_time = np.std(frame_times)
                logging.info(f"Frame timing: Mean={mean_frame_time*1000:.1f}ms, Std={std_frame_time*1000:.1f}ms")
                win.frameIntervals = []  # Reset for next trial
        
        logging.info(f"Completed trial {trial_num}/{len(trials)}: {hand} hand")
        
        # Check for quit
        if event.getKeys(keyList=["escape"]):
            break

    # Cleanup
    win.close()
    logging.info("Neurofeedback experiment finished.")