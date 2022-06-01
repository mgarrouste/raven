# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on April 6, 2021

@author: talbpaul
"""
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as tic
import numpy as np
import pandas as pd

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes

class OptPath(PlotInterface):
  """
    Plots the path that variables took during an optimization, including accepted and rejected runs.
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""The name of the RAVEN DataObject from which the data should be taken for this plotter.
              This should be the SolutionExport for a MultiRun with an Optimizer."""))
    spec.addSub(InputData.parameterInputFactory('vars', contentType=InputTypes.StringListType,
        descr=r"""Names of the variables from the DataObject that will be plotted on the optimization path plot."""))
    varUnitsNode = InputData.parameterInputFactory('varUnits',
        descr=r"""Units of the variables from the DataObject that will be plotted on the optimization path plot.""")
    varNode = InputData.parameterInputFactory('var', contentType=InputTypes.StringType,
        descr="""Name of the variable that unit is begin defined.""" )
    varNode.addParam("unit", InputTypes.StringType, True, descr="""Unit of the variable""")
    varUnitsNode.addSub(varNode)
    spec.addSub(varUnitsNode)
    return spec

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'OptPath Plot'
    self.source = None      # reference to DataObject source
    self.sourceName = None  # name of DataObject source
    self.vars = None        # variables to plot
    self.varUnits = {}      # units of variables to plot
    self.markerMap = {'first': 'o',
                      'accepted': 'o',
                      'rejected': 'x',
                      'rerun': '.',
                      'final': 'o'}
    self.colorMap = {'first': 'y',
                      'accepted': 'g',
                      'rejected': 'r',
                      'rerun': 'c',
                      'final': 'm'}
    self.markers = defaultdict(lambda: 'k.', self.markerMap)

  def handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    super().handleInput(spec)
    self.sourceName = spec.findFirst('source').value
    self.vars = spec.findFirst('vars').value
    # create dic ngcc_capacity: unit
    vU = spec.findFirst('varUnits')
    for i in vU.findAll('var'):
      self.varUnits[i.value]= i.parameterValues['unit']
    # checker; this should be superceded by "required" in input params
    if self.sourceName is None:
      self.raiseAnError(IOError, "Missing <source> node!")
    if self.vars is None:
      self.raiseAnError(IOError, "Missing <vars> node!")

  def initialize(self, stepEntities):
    """
      Function to initialize the OutStream. It basically looks for the "data"
      object and links it to the system.
      @ In, stepEntities, dict, contains all the Objects are going to be used in the
                                current step. The sources are searched into this.
      @ Out, None
    """
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" was found in the Step for SamplePlot "{self.name}"!')
    self.source = src
    # sanity check
    dataVars = self.source.getVars()
    missing = [var for var in (self.vars+['accepted']) if var not in dataVars]
    if missing:
      msg = f'Source DataObject "{self.source.name}" is missing the following variables ' +\
            f'expected by OptPath plotter "{self.name}": '
      msg += ', '.join(f'"{m}"' for m in missing)
      self.raiseAnError(IOError, msg)

  def run(self):
    """
      Main run method.
      @ In, None
      @ Out, None
    """
    fig, axes = plt.subplots(len(self.vars), 1, sharex=True)
    axes[-1].set_xlabel("Optimizer Iteration")
    data =[]
    for r in range(len(self.source)): # realizations
      data.append(self.source.realization(index=r, asDataSet=True, unpackXArray=False))
    rlz = pd.DataFrame(data)
    dfa = rlz.query("accepted in ['first', 'accepted']")
    dfr = rlz.query("accepted not in ['first', 'accepted']")
    for var, ax in zip(self.vars, axes):
        # Sci. notation for everything > 1000.
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 3))
        # Title Case Vars, but don't change abbreviations...
        title = " ".join([i if i.isupper() else i.title() for i in var.split("_")])
        ax.set_title(title)
        try:
          unit = self.varUnits[var] # Capacity unit
          ylabel = "\n(${}$)".format(unit)
          ax.set_ylabel(ylabel)
        except KeyError:
          pass          
        for k, d in dfr.groupby("accepted"):
            y = np.abs(d[var].to_numpy()) 
            #if var == 'mean_NPV':
            #    y = d[var].to_numpy() - scalar
            ax.scatter(
                d.iteration.to_numpy(),
                y,
                c=d.accepted.map(self.colorMap),
                label=k,
                marker="o",
                alpha=0.8,
            )
        y = np.abs(dfa[var].to_numpy()) 
        if var == 'mean_NPV':
            y = dfa[var].to_numpy() 
        ax.plot(
            dfa.iteration.to_numpy(),
            y,
            label="accepted",
            color=self.colorMap["accepted"],
            marker=self.markerMap["accepted"],
            linestyle="-",
        )
        if 'storage' in var:
            yabs_max = abs(max(ax.get_ylim(), key=abs)) + 0.10
            ax.set_ylim(bottom=-0.10, top=yabs_max)
            formatter = tic.StrMethodFormatter('{x:.2f}')
            ax.yaxis.set_major_formatter(formatter)
        ax.grid()
    """ for r in range(len(self.source)): # realizations
      rlz = self.source.realization(index=r, asDataSet=True, unpackXArray=False)
      dfa = rlz.query("accepted in ['first', 'accepted']")
      dfr = rlz.query("accepted not in ['first', 'accepted']")
      accepted = rlz['accepted']
      for v, var in enumerate(self.vars):
        ax = axes[v]
        value = rlz[var]
        #self.addPoint(ax, r, value, accepted)
        # Fix y label
        ylabel = str(var).replace("_capacity","")
        ylabel = " ".join([i if i.isupper() else i.title() for i in ylabel.split("_")])
        try:
          unit = self.varUnits[var] # Capacity unit
          ylabel += "\n(${}$)".format(unit)
        except KeyError:
          pass
        ax.set_ylabel(ylabel)
        ax.grid() """
    # common legend
    fig.subplots_adjust(right=0.80)
    # Set middle axes to have legend to the right of the plot.
    # Also reorder legend to have 'accepted' appear on top.
    handles, labels = axes[-1].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    lg = fig.legend(handles, labels,
               loc='center right',
               borderaxespad=0.1,
               markerscale=1.2,
               bbox_to_anchor=(1, 0.5),
               title='Legend')
    # Only integer values for x axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Space adjustment
    plt.setp(lg.get_title(), multialignment="center")
    plt.subplots_adjust(hspace=0.1)
    fig.tight_layout()
    # Have to update canvas to get actual legend width
    fig.canvas.draw()
    # The following will place the legend in a non-weird place
    fig.subplots_adjust(right=self.get_adjust(lg.get_frame().get_width()))
    plt.savefig(f'{self.name}.png')

  def get_adjust(self, width):
    if width < 149:
        return 0.78
    elif width < 164:
        return 0.73
    elif width < 169:
        return 0.73
    else:
        return 0.7

  def addPoint(self, ax, i, value, accepted):
    """
      Plots a point in the optimization path.
      @ In, ax, pyplot axis, axis to plot on
      @ In, i, int, iteration number
      @ In, value, float, variable value
      @ In, accepted, str, acceptance condition
      @ Out, lines, list, lines created
    """
    lines = ax.plot(i, value, f'{self.markers[accepted]}')
    return lines
