#!/bin/bash

# Define the range of folds and the list of num_classes
num_classes=(2 3 4 5 6 7)
yaml_file="modelling/stage_classifier/config.yaml"  # Path to your YAML file

# Function to update the YAML file
update_yaml() {
    python -c "
import yaml

# Load YAML file
with open('$yaml_file', 'r') as file:
    config = yaml.safe_load(file)

# Update values
config['Stage_classifier']['num_classes'] = int('$1')  # Convert to integer

# Save updated YAML file
with open('$yaml_file', 'w') as file:
    yaml.safe_dump(config, file)
    " $1
}


runner_file="modelling/stage_classifier/trainer.py"  # Path to your Python script

# Iterate over each combination of fold and num_classes
for num_class in "${num_classes[@]}"; do
    echo "Updating YAML and running $runner_file with num_classes=$num_class"

    # Update the YAML file with current parameters
    update_yaml $num_class

    # Run the Python script and wait for it to complete
    python $runner_file
    if [ $? -ne 0 ]; then
        echo "Error occurred while running the trainer with num_classes=$num_class"
        exit 1
    fi

    echo "Completed num_classes=$num_class"
done

echo "All combinations have been processed."
