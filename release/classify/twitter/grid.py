from numpy import prod
from collections import Counter
from zutils.algo.utils import format_list_to_string
from zutils.algo.utils import ensure_directory_exist


class GridSpace:
    def __init__(self, ranges, bins):
        self.ranges = ranges
        self.num_bins = bins
        self.raw_cell_freq_counter = Counter()  # raw_cell_id -> freq
        self.new_to_old = {}  # new_cell_id -> raw_cell_id
        self.old_to_new = {}  # raw_cell_id -> new_cell_id

    # encode the cells by frequency, so that the most frequent cell has id 0
    def encode_cells(self, points, min_freq=1):
        # count the cells from the points
        for point in points:
            raw_cell_id = self.get_raw_cell_id(point)
            self.raw_cell_freq_counter[raw_cell_id] += 1
        # ranked list, (old_cell_id, freq)
        ranked_cells = self.raw_cell_freq_counter.most_common()
        # build mappings
        new_id = 0
        for (old_id, count) in ranked_cells:
            # do not include the cells appear less than min_freq times
            if count < min_freq:
                break
            self.new_to_old[new_id] = old_id
            self.old_to_new[old_id] = new_id
            new_id += 1

    # the total number of cells
    def num_cells(self):
        return int(prod(self.num_bins))

    def num_nonempty_cells(self):
        return len(self.new_to_old)

    # get the converted cell id of a point
    def get_cell_id(self, point):
        raw_cell_id = self.get_raw_cell_id(point)
        ret = None if raw_cell_id not in self.old_to_new else self.old_to_new[raw_cell_id]
        return ret

    # get the number of points in a cell
    def get_cell_count(self, cell_id):
        return self.raw_cell_freq_counter[self.new_to_old[cell_id]]

    # get the raw cell id of a point
    def get_raw_cell_id(self, point):
        index_list = self.calc_index_along_axis(point)
        factor = 1
        res = 0
        for i, index in enumerate(index_list):
            res += index * factor
            factor *= self.num_bins[i]
        return res

    def calc_index_along_axis(self, point):
        res = []
        for i, value in enumerate(point):
            # the range for each axis
            begin, end = self.ranges[i]
            num_bin = self.num_bins[i]
            index = int(num_bin * (value - begin) / (end - begin))
            # when value == end, need to decrease the index
            if index >= num_bin:    index = num_bin - 1
            res.append(index)
        return res

    # get the cell center location of a given cell id
    def cell_id_to_location(self, cell_id):
        raw_cell_id = self.new_to_old[cell_id]
        return self.raw_cell_id_to_location(raw_cell_id)

    # get the cell center location of a given raw cell id
    def raw_cell_id_to_location(self, cell_id):
        axis_index_list = []
        self.calc_index_list_from_id(cell_id, self.num_bins, axis_index_list)
        ret = []
        for i, index in enumerate(axis_index_list):
            axis_min, axis_max = self.ranges[i]
            n_bin = self.num_bins[i]
            grid_width = (axis_max - axis_min) / n_bin
            ret.append(axis_min + grid_width * (index + 0.5))
        return ret

    def calc_index_list_from_id(self, cell_id, bins, res):
        if len(bins) == 1:
            res.insert(0, cell_id)
            return
        current_axis_index = cell_id / int(prod(bins[:-1]))
        res.insert(0, current_axis_index)
        cell_id -= current_axis_index * int(prod(bins[:-1]))
        return self.calc_index_list_from_id(cell_id, bins[:-1], res)

    # write (new_id -> location) into a file
    def write_to_file(self, grid_file):
        ensure_directory_exist(grid_file)
        n_grid = len(self.new_to_old)
        with open(grid_file, 'w') as fout:
            for cell_id in xrange(n_grid):
                center = self.cell_id_to_location(cell_id)
                count = self.get_cell_count(cell_id)
                fout.write(format_list_to_string([cell_id, center, count]) + '\n')


if __name__ == '__main__':
    gs = GridSpace([(0, 100), (0, 100)], [10, 10])
    print gs.get_raw_cell_id([90, 1])
    print gs.num_cells()
    print gs.raw_cell_id_to_location(9)
