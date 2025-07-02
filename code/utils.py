import json
import pickle
from os.path import join
import os

import keras


class C:
    """
    Constants class
    """

    SMALL = "small"
    MEDIUM = "medium"
    BIG = "big"

    FONT = "font"
    AXES_TITLE = "axes title"
    AXES_LABEL = "axes label"
    XTICK = "xtick"
    YTICK = "ytick"
    LEGEND_FONT = "legend font"
    LEGEND_TITLE = "legend title"
    TITLE = "title"
    LATEX = "latex"

    COLUMN = 0
    ROW = 1

    MAX = 0
    MIN = 1

    STYLES = ["-", "--"]
    COLORS = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:olive",
        "tab:purple",
    ]
    MARKERS = ["v", "^", "s", "o", "*", "h"]

    DEFAULT_COLOR = "tab:blue"
    DEFAULT_MARKER = None
    DEFAULT_STYLE = "-"

    SAVE_OK_MSG = "{0} saved in {1}"
    LOAD_OK_MSG = "{0} loaded from {1}"

    WARNING_MSG = "Could not find {0}. Using default {1}"
    TICKS_WARNING_MSG = 'The number of {0}ticks labels is different from the number of {0}ticks'

    LOAD_ERROR_MSG = (
        "Could not load {0} from {1}."
    )
    SAVE_ERROR_MSG = "Could not save {0} in {1}"
    DATASET_ERROR_MSG = (
        "Error while parsing the dataset. Make sure dataset is a matrix."
    )
    SIZE_ERROR_MSG = (
        f"Unknown size value. Accepted sizes are {SMALL}, {MEDIUM} and {BIG}"
    )
    SIZE_LATEX_ERROR_MSG = SIZE_ERROR_MSG + f"\nlatex accepted sizes are true and false"

    DIMENSIONS_ERROR_MSG = f"Unknown dimension value. Accepted dimensions are {0}"
    VALUE_ERROR_MSG = "New value must be an integer"
    SMALL_ERROR_MSG = "Small size must be between 0 and medium size ({0})"
    MEDIUM_ERROR_MSG = "Medium size must be between small size ({0}) and big size ({1})"
    BIG_ERROR_MSG = "Big size must be greater than medium size ({0})"

    X = 'x'
    Y = 'y'

    NONE = 0
    JSON = 1
    TXT = 2
    PICKLE = 3
        
def _load_file(
    read_file: str,
    load_ok: str = "File loaded",
    error: str = f"Error while loading file",
    type: int = C.NONE,
) -> object:
    """
    Load single file. It handles txt, pddl, json and pickle files

    Args:
        read_file: string that contains the path to the file
        load_ok: string that contains the message to print when the loading is successful
        error: string that contains the message to print when the loading is not successful

    Returns:
        the loaded file
    """
    try:
        if type == C.JSON or (type == C.NONE and read_file.lower().endswith(".json")):
            with open(read_file, "r") as rf:
                o = json.load(rf)
        elif type == C.TXT or (
            type == C.NONE
            and (
                read_file.lower().endswith(".txt")
                or read_file.lower().endswith(".pddl")
            )
        ):
            with open(read_file, "r") as rf:
                o = rf.readlines()
        else:
            with open(read_file, "rb") as rf:
                o = pickle.load(rf)
        print(load_ok)
    except FileNotFoundError:
        print(error)
        o = None

    return o


def load_from_folder(read_dir: str, files: list, type: int = 0) -> list:
    """
    Load files from a given folder. Supports txt, pddl, json and pickle files.

    Args:
        read_dir: a string that contains the path to a folder
        files: a list of file names within the folder

    Returns:
        A list of loaded files
    """

    to_return = []
    for file_name in files:
        to_return.append(
            _load_file(
                join(read_dir, file_name),
                load_ok=C.LOAD_OK_MSG.format(file_name, read_dir),
                error=C.LOAD_ERROR_MSG.format(file_name, read_dir),
                type=type,
            )
        )
    return to_return


def save_file(o: object, target_dir: str, filename: str) -> bool:
    """
    Saves a given object in a file. Supports txt, json and pickle files.
    Args:
        o: object to save
        target_dir: path to the target directory. It is created if it does not exist
        filename: target file name. If needed it must contain the extension

    Returns:
        True if the saving is successful, False otherwise
    """

    os.makedirs(target_dir, exist_ok=True)
    try:
        if filename.endswith(".json") or filename.endswith(".JSON"):
            with open(join(target_dir, filename), "w") as wf:
                json.dump(o, wf, indent=4)
        elif filename.endswith(".txt") or filename.endswith(".TXT"):
            with open(join(target_dir, filename), "w") as wf:
                wf.writelines(o)
        else:
            with open(join(target_dir, filename), "wb") as wf:
                pickle.dump(o, wf)
        wf.close()
        print(C.SAVE_OK_MSG.format(filename, target_dir))
        return True
    except pickle.PicklingError:
        print(C.SAVE_ERROR_MSG.format(filename, target_dir))
        return False



from keras import backend as K
import tensorflow as tf
from keras.layers import Layer
from keras import initializers, regularizers, constraints

@keras.saving.register_keras_serializable()
class AttentionWeights(Layer):
    def __init__(
        self,
        step_dim,
        W_regularizer=None,
        b_regularizer=None,
        W_constraint=None,
        b_constraint=None,
        bias=True,
        **kwargs
    ):
        self.supports_masking = True
        self.init = initializers.get("glorot_uniform")
        # self.init = initializers.get(Constant(value=1))

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(AttentionWeights, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(
            shape=(input_shape[-1],),
            initializer=self.init,
            name="{}_W".format(self.name),
            regularizer=self.W_regularizer,
            constraint=self.W_constraint,
        )
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(
                shape=(input_shape[1],),
                initializer="zero",
                name="{}_b".format(self.name),
                regularizer=self.b_regularizer,
                constraint=self.b_constraint,
            )
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        # todo is K.dot equal to keras.layers.dot as keras_dot?
        eij = tf.reshape(
            tf.matmul(
                tf.reshape(x, (-1, features_dim)), tf.reshape(self.W, (features_dim, 1))
            ),
            (-1, step_dim),
        )

        if self.bias:
            eij += self.b

        eij = tf.tanh(eij)

        a = tf.exp(eij)

        if mask is not None:
            a *= tf.cast(mask, K.floatx())

        a /= tf.cast(tf.reduce_sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx()) #todo is K.sum equal to tf.reduce_sum?

        return a

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

    def get_config(self):
        config = {"step_dim": self.step_dim}
        base_config = super(AttentionWeights, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

@keras.saving.register_keras_serializable()
class ContextVector(Layer):
    def __init__(self, **kwargs):
        super(ContextVector, self).__init__(**kwargs)
        self.features_dim = 0

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.features_dim = input_shape[0][-1]
        self.built = True

    def call(self, x, **kwargs):
        assert len(x) == 2
        h = x[0]
        a = x[1]
        a = tf.expand_dims(a, axis=-1)
        weighted_input = h * a
        return tf.reduce_sum(weighted_input, axis=1) #todo is K.sum equal to tf.reduce_sum?

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.features_dim

    def get_config(self):
        base_config = super(ContextVector, self).get_config()
        return dict(list(base_config.items()))
