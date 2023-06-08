import config
import honeypot
import ml_model
import network_interaction

def main():
    # Setup configuration
    config.load_config()

    # Start the network interactions
    network_interaction.start_network_interaction()

    # Create the honeypots
    honeypots = honeypot.create_honeypots()

    # Start the honeypots
    honeypot.start_honeypots(honeypots)

    # Train the machine learning model
    x_train, y_train = ml_model.load_data()
    ml_model.train_model(x_train, y_train)

    # Monitor network traffic
    network_interaction.monitor_traffic()

if __name__ == "__main__":
    main()
