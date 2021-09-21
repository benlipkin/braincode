import ast
import json
import os
import pickle as pkl
import subprocess
from io import BytesIO
from tokenize import tok_name, tokenize

import numpy as np


class ProgramBenchmark:
    def __init__(self, benchmark, basepath, fnames):
        self._benchmark = benchmark
        self._base_path = basepath
        self._fnames = fnames

    def fit_transform(self, programs):
        outputs = []
        for i, program in enumerate(programs):
            metrics = ProgramMetrics(program, str(self._fnames[i]), self._base_path)
            if self._benchmark == "task-lines":
                metric = metrics.get_number_of_runtime_steps()
            elif self._benchmark == "task-nodes":
                metric = metrics.get_ast_node_counts()
            elif self._benchmark == "task-tokens":
                metric = metrics.get_token_counts()
            else:
                complexity = metrics.get_halstead_complexity_metrics()
                if self._benchmark == "task-halstead":
                    metric = complexity["program_length"]
                elif self._benchmark == "task-cyclomatic":
                    metric = complexity["cyclomatic_complexity"]
                else:
                    raise ValueError(
                        "Undefined program metric. Make sure to use valid identifier."
                    )
            outputs.append(metric)
        return np.array(outputs).reshape([-1, 1])


class ProgramMetrics:
    def __init__(self, program, path, base_path):
        self.program = program
        self.path = path
        self.base_path = base_path
        self.outpath = os.path.join(self.base_path, ".cache", "profiler")
        os.makedirs(self.outpath, exist_ok=True)
        self.fname = "_".join(self.path.split(os.sep)[-2:])

        # Prepare a copy of the src for the profilers
        if not os.path.exists(os.path.join(self.outpath, self.fname)):
            src = self._prepare_src_for_profiler(program)
            with open(os.path.join(self.outpath, self.fname), "w") as fp:
                fp.write(src)

    def get_token_counts(self):
        exclude_tokens_types = [
            "NEWLINE",
            "NL",
            "INDENT",
            "DEDENT",
            "ENDMARKER",
            "ENCODING",
        ]
        exclude_ops = ["[", "]", "(", ")", ",", ":"]
        token_count = 0
        for res in tokenize(BytesIO(self.program.encode("utf-8")).readline):
            if res and tok_name[res.type] not in exclude_tokens_types:
                if (
                    tok_name[res.type] == "OP" and res.string not in exclude_ops
                ) or tok_name[res.type] != "OP":
                    token_count += 1
        return token_count

    def get_ast_node_counts(self):
        root = ast.parse(self.program)
        ast_node_count = 0
        for _ in ast.walk(root):
            ast_node_count += 1
        return ast_node_count

    def get_halstead_complexity_metrics(self, sec=30):
        # See https://radon.readthedocs.io/en/latest/intro.html for details on Halstead metrics
        # reported by Radon

        local_fname = os.path.join(self.outpath, self.fname)
        cmd = ["radon", "hal", local_fname, "-j"]
        try:
            output = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=sec
            )
            out = output.stdout.decode("utf-8")
            out = json.loads(out)
            if not out:
                err = output.stderr.decode("utf-8")
                print(err)
        except Exception as e:
            print(e)

        metrics = {}
        metrics["number_of_distinct_operators"] = out[local_fname]["total"][0]
        metrics["number_of_distinct_operands"] = out[local_fname]["total"][1]
        metrics["number_of_operators"] = out[local_fname]["total"][2]
        metrics["number_of_operands"] = out[local_fname]["total"][3]
        metrics["program_length"] = out[local_fname]["total"][6]
        metrics["program_difficulty"] = out[local_fname]["total"][7]
        metrics["program_effort"] = out[local_fname]["total"][8]

        cmd = ["radon", "cc", local_fname, "-j"]
        try:
            output = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=sec
            )
            out = output.stdout.decode("utf-8")
            out = json.loads(out)
            if not out:
                err = output.stderr.decode("utf-8")
                print(err)
        except Exception as e:
            print(e)

        metrics["cyclomatic_complexity"] = out[local_fname][0]["complexity"]
        # print(json.dumps(metrics, indent=2))
        return metrics

    @staticmethod
    def _prepare_src_for_profiler(src):
        indent = "  "
        src = src.replace("\n", "\n" + indent)
        src = indent + src
        src = "@profile\ndef profile_me():\n" + src + "\nprofile_me()"
        return src

    def get_number_of_runtime_steps(self, sec=30):
        """
        Requires the package line_profiler to be installed.
        Picks up the # hits for every line from the output of this profiler.
        See https://github.com/rkern/line_profiler

        :param sec: Timeout for subprocess.run
        :return:[# of lines] executed
        """
        if not self.path[-3:] == ".py":
            raise ValueError("Unrecognized file type")
        
        if not os.path.exists(os.path.join(self.outpath, self.fname + ".lprof")):
            cmd = [
                "kernprof",
                "-o",
                os.path.join(self.outpath, self.fname + ".lprof"),
                "-l",
                os.path.join(self.outpath, self.fname),
            ]
            try:
                output = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=sec
                )
                out = output.stdout.decode("utf-8")
                if not out:
                    err = output.stderr.decode("utf-8")
                    print(err)
            except Exception as e:
                print(e)

        sum_hits = 0
        with open(os.path.join(self.outpath, self.fname + ".lprof"), "rb") as fp:
            obj = pkl.load(fp)
            if len(obj.timings) > 1:
                print("something not right")
            else:
                # obj.timings format - {filename: [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]}
                # x1 - line number, y1 - hits, z1 - time spent on the line
                for v in obj.timings.values():
                    for i in v:
                        # index 1 contains number of hits.
                        sum_hits += i[1]
        return sum_hits
