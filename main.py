import tensorflow as tf
from src.SmartOffiseOrchestrator import SmartOfficeOrchestrator


graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)


if __name__ == "__main__":
    orchestrator = SmartOfficeOrchestrator("files", sess)
    while True:
        orchestrator.run()
