# This class inherits all behaviour from a ModelCheckpoint.
# One of the overriden methods are: on_epoch_end(), due to having to increment the epoch number.
# The epoch number is not tracked due to having an outer loop around the model.fit_generator train function.
# The best models based on valdiation loss and the validation metric are stored into a yaml file.
# The .yaml file keeps track of which epoch reached the best validation loss and validation metric value.
# The private overriden method _save_model() has one change, being the ._update_yaml() call.



from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.platform import tf_logging as logging
import yaml


# Override modelcheckpoint callback
class ModelCheckpointYaml(ModelCheckpoint):
    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 save_freq='epoch',
                 options=None,
                 mc_dict_filename="mc_dict_filename.yaml",
                 **kwargs):
        super(ModelCheckpointYaml, self).__init__(
            filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq,
            options=options
        )
        self.mc_dict_filename = mc_dict_filename
        self.mc_dict = {"epoch": 0, self.monitor: -1}
        self.epoch = 0


    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        # pylint: disable=protected-access
        if self.save_freq == 'epoch':
            self._save_model(epoch=epoch, logs=logs)
        self.epoch += 1


    def _save_model(self, epoch, logs):
        """Saves the model.
        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if isinstance(self.save_freq,
                    int) or self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                    'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                        current, filepath))
                        self.best = current
                        self._update_yaml()
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' %
                        (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

            self._maybe_remove_file()


    # Store epoch and score/score in a yaml file if the score/score is better
    def _update_yaml(self):
        if self.best == float('inf'):
            return 
        self.mc_dict["epoch"] = self.epoch
        self.mc_dict[self.monitor] = float(self.best)

        f = open(self.mc_dict_filename, "w")
        yaml.dump(self.mc_dict, f)
        f.close()