SESSION_NAME="multi_car"

ENV_SETUP="export XLA_PYTHON_CLIENT_PREALLOCATE=false; export JAX_PLATFORM_NAME=cpu; export XLA_FLAGS='--xla_force_host_platform_device_count=1'; export OMP_NUM_THREADS=1"

echo "Allow about 50 seconds in total for all windows to start"

EPISODES_PER_WINDOW=25
START=1

# Use unified exp_name for all windows
EXP_NAME="alpha_racer_multi"

# Start the first window
END=$((START + EPISODES_PER_WINDOW - 1))
echo "Creating window data0 [$START-$END]"
tmux new-session -d -s $SESSION_NAME -n "data0" \
"bash -c '$ENV_SETUP; python3 multi_car_blocking.py --exp_name $EXP_NAME --start_ep $START --end_ep $END --use_wandb; echo \"[data0 done] Press any key to exit...\"; read'"

sleep 5

for i in {1..9}
do
    START=$((END + 1))
    END=$((START + EPISODES_PER_WINDOW - 1))

    if [ $END -gt 242 ]; then
        END=242
    fi

    echo "Creating window data$i [$START-$END]"
    tmux new-window -t $SESSION_NAME -n "data$i" \
    "bash -c '$ENV_SETUP; python3 multi_car_blocking.py --exp_name $EXP_NAME --start_ep $START --end_ep $END --use_wandb; echo \"[data${i} done] Press any key to exit...\"; read'"

    if [ $END -eq 242 ]; then
        break
    fi

    sleep 5
done

tmux attach -t $SESSION_NAME