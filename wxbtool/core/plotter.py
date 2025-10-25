from wxbtool.norms.meanstd import denormalizors
from wxbtool.util.plot import plot_image


class Plotter:
    def __init__(self, model, climatology_accessors):
        self.model = model
        self.climatology_accessors = climatology_accessors

    def plot_date(self, data, variables, span, key):
        for var in variables:
            item = data[var]
            for ix in range(span):
                if item.dim() == 4:
                    height, width = item.size(-2), item.size(-1)
                    dat = item[0, ix].detach().cpu().numpy().reshape(height, width)
                else:
                    height, width = item.size(-2), item.size(-1)
                    dat = item[0, 0, ix].detach().cpu().numpy().reshape(height, width)
                self.model.artifacts[f"{var}_{ix:02d}_{key}"] = {"var": var, "data": dat}

    def plot_map(self, inputs, targets, results, indexes, batch_idx, mode, path):
        if inputs[self.model.model.setting.vars_out[0]].dim() == 4:
            zero_slice = 0, 0
        else:
            zero_slice = 0, 0, 0

        for bas, var in enumerate(self.model.model.setting.vars_out):
            input_data = inputs[var][zero_slice].detach().cpu().numpy()
            truth = targets[var][zero_slice].detach().cpu().numpy()
            forecast = results[var][zero_slice].detach().cpu().numpy()
            input_data = denormalizors[var](input_data)
            forecast = denormalizors[var](forecast)
            truth = denormalizors[var](truth)
            plot_image(
                var,
                input_data=input_data,
                truth=truth,
                forecast=forecast,
                title=var,
                year=self.climatology_accessors[mode].yr_indexer[indexes[0]],
                doy=self.climatology_accessors[mode].doy_indexer[indexes[0]],
                save_path="%s/%s_%02d.png" % (path, var, batch_idx),
            )
