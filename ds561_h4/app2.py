from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1

def banned_countries_message():

    timeout = 5.0
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path('jacks-project-398813', 'banned_countries-sub')

    def callback(message: pubsub_v1.subscriber.message.Message) -> None:
        print(f"Received {message}.")
        message.ack()

    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print(f"Listening for messages on {subscription_path}..\n")

    # Wrap subscriber in a 'with' block to automatically call close() when done.
    with subscriber:
        try:
            # When `timeout` is not set, result() will block indefinitely,
            # unless an exception is encountered first.
            streaming_pull_future.result(timeout=timeout)
        except TimeoutError:
            streaming_pull_future.cancel()  # Trigger the shutdown.
            streaming_pull_future.result()  # Block until the shutdown is complete.

banned_countries_message()
print("The above requests were intercepted and denied")