from datetime import datetime
import json



def print_arguments(args, parser):
    """
    Prints all arguments used in the script and highlights whether default values were used.
    
    Args:
        args: The parsed arguments (from `parser.parse_args()`).
        parser: The `argparse.ArgumentParser` object.
    """
    print("Arguments used in this script:")
    print("\n" )
    
    # Get the default values from the parser
    defaults = {action.dest: action.default for action in parser._actions if action.dest != "help"}
    
    # Iterate through the arguments and print their values
    for arg_name, arg_value in vars(args).items():
        if arg_name in defaults:
            default_value = defaults[arg_name]
            if arg_value == default_value:
                # Highlight if the default value was used
                print(f"{arg_name}: {arg_value} (default)")
            else:
                # Highlight if the user provided a value
                print(f"{arg_name}: {arg_value} (user-provided)")
        else:
            # Handle cases where the argument doesn't have a default (e.g., --kwargs)
            print(f"{arg_name}: {arg_value}")

    print("-" * 50)



def save_training_metadata(args, best_epoch, score, percentage):
    """
    Save all the parameters, hyper-parameters, and codebook usage in a JSON file.

    Args:
        args: The parsed command-line arguments (from argparse).
        model: The deep learning model.
        codebook_usage: A dictionary or list containing codebook usage information (optional).
    """


    # Get the current timestamp for the filename
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = (args.model_name).replace('.pth', '.json')

        # Parse kwargs arguments into a dictionary
    kwargs_dict = {}
    if args.kwargs:
        for kwarg in args.kwargs:
            if '=' in kwarg:
                key, value = kwarg.split('=', 1)
                kwargs_dict[key] = value


    # Prepare the metadata dictionary
    metadata = {
        "model_parameters": {
            "embedding_dim": args.D,
            "num_embeddings": args.K,
            "downsampling_factor": args.downsampling_factor,
            "residual": args.use_residual,
            "num_quantizers": args.num_quantizers,
            "shared_codebook": args.shared_codebook,
            "beta": args.beta,
            "decay": args.decay,
            "data_mod": args.data_mod,
            "loss_func": args.loss_func,
        },
        "kwargs_arguments": kwargs_dict,  # Store additional kwargs arguments
        "evaluation": {
            "best_epoch" : best_epoch, 
            "score" : score,
            "codebook_usage" : percentage,
        },
        "timestamp": timestamp,
    }

    # Save the metadata to a JSON file
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"( Training metadata saved to {filename} )")


def save_training_metadata_Refit(args, model_parameters, best_epoch, score, percentage):
    """
    Save all the parameters, hyper-parameters, and codebook usage in a JSON file.

    Args:
        args: The parsed command-line arguments (from argparse).
        model: The deep learning model.
        codebook_usage: A dictionary or list containing codebook usage information (optional).
    """


    # Get the current timestamp for the filename
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = (args.model_name).replace('.pth', '.json')

        # Parse kwargs arguments into a dictionary
    kwargs_dict = {}
    if args.kwargs:
        for kwarg in args.kwargs:
            if '=' in kwarg:
                key, value = kwarg.split('=', 1)
                kwargs_dict[key] = value


    # Prepare the metadata dictionary
    metadata = {
        "baseline_model" : args.baseline_model_name,
        "model_parameters" : model_parameters,
        "kwargs_arguments": kwargs_dict,  # Store additional kwargs arguments
        "evaluation": {
            "best_epoch" : best_epoch, 
            "score" : score,
            "codebook_usage" : percentage,
        },
        "timestamp": timestamp,
    }

    # Save the metadata to a JSON file
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"( Training metadata saved to {filename} )")



def load_model_metadata(model_path):

    json_filepath = model_path.replace('.pth', '.json')
    # Load the JSON file
    with open(json_filepath, 'r') as f:
        metadata = json.load(f)

    # Extract model parameters from the metadata
    model_params = metadata.get("model_parameters", {})

    return model_params