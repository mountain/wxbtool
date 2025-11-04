from wxbtool.norms.meanstd import denormalizors


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
                    dat = item[0, ix].detach().float().cpu().numpy().reshape(height, width)
                else:
                    height, width = item.size(-2), item.size(-1)
                    dat = item[0, 0, ix].detach().float().cpu().numpy().reshape(height, width)
                self.model.artifacts[f"{var}_{ix:02d}_{key}"] = {"var": var, "data": dat, "type": "data", "kind": key}

    def plot_map(self, inputs, targets, results, indexes, mode):
        if inputs[self.model.model.setting.vars_out[0]].dim() == 4:
            zero_slice = 0, 0
        else:
            zero_slice = 0, 0, 0

        for bas, var in enumerate(self.model.model.setting.vars_out):
            input_data = inputs[var][zero_slice].detach().float().cpu().numpy()
            truth = targets[var][zero_slice].detach().float().cpu().numpy()
            forecast = results[var][zero_slice].detach().float().cpu().numpy()
            input_data = denormalizors[var](input_data)
            forecast = denormalizors[var](forecast)
            truth = denormalizors[var](truth)

            year = self.climatology_accessors[mode].yr_indexer[indexes[0]]
            doy = self.climatology_accessors[mode].doy_indexer[indexes[0]]

            self.model.artifacts[f"{var}_{year:4d}_{doy:03d}_input"] = {"var": var, "data": input_data, "year": year, "doy": doy, "type": "map", "kind": "input"}
            self.model.artifacts[f"{var}_{year:4d}_{doy:03d}_truth"] = {"var": var, "data": truth, "year": year, "doy": doy, "type": "map", "kind": "truth"}
            self.model.artifacts[f"{var}_{year:4d}_{doy:03d}_forecast"] = {"var": var, "data": forecast, "year": year, "doy": doy, "type": "map", "kind": "forecast"}
