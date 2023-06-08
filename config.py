def load_config():
    # Load configuration from file or set default values
    config_values = {}

    # Load configuration from a file
    try:
        with open("config.txt", "r") as file:
            for line in file:
                key, value = line.strip().split("=")
                config_values[key] = value
        print("Configuration loaded from file.")
    except FileNotFoundError:
        print("Configuration file not found. Using default values.")

    # Set default values if not loaded from file
    if not config_values:
        config_values = {
            "port_numbers": [80, 443],
            "response_delay": 0.5,
            "logging_level": "DEBUG",
            "vulnerabilities": ["SQL Injection", "Cross-Site Scripting"],
        }

    return config_values
