from model.dynamics import Dynamic


class Linear(Dynamic):
    def init_constants(self):
        pass

    def integrate_samples(self, v, x):
        return v

    def integrate_distribution(self, v_dist, x):
        return v_dist