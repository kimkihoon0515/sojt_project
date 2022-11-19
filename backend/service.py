import bentoml
import numpy as np

mnist_runner = bentoml.mlflow.get('mnist_clf:latest').to_runner()

svc = bentoml.Service('mnist_clf', runners=[ mnist_runner ])

input_spec = bentoml.io.NumpyNdarray(
    dtype="float32",
    shape=[-1,784],
    enforce_shape=True,
    enforce_dtype=True,
)

@svc.api(input=input_spec, output=bentoml.io.NumpyNdarray())
def predict(input_arr):
    result = mnist_runner.predict.run(input_arr)

    answer = []
    for i in result:
        answer.append(np.argmax(i))

    answer = np.array(answer)
    return answer