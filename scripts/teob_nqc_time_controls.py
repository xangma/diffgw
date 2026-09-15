#!/usr/bin/env python3
"""Separate NQC evaluation-time changes from derivative-policy changes.

This is a frozen-input diagnostic. Recovered coefficients never enter runtime.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from teob_continuous_fd_attribution import OUT, json_value
from torchwave.approximants.eob.numerical_attachment import natural_cubic_spline
from torchwave.approximants.eob.teob_merger_nqc_clean import matching_index

def main():
    report={}
    with torch.no_grad():
        for name in ('catalog','bbh_reference','equal_bbh','unequal_bns'):
            folder=OUT/name; path=folder/'continuous.npz'; d=np.load(path)
            prior=json.loads((folder/'matching-controls.json').read_text())['variants']['grid']
            run=json.loads((folder/'continuous.json').read_text())
            tensor=lambda key:torch.as_tensor(d[key],dtype=torch.float64)
            time=tensor('nqc_original_time'); phase=tensor('nqc_original_phase')
            aux=tensor('nqc_auxiliary_time'); freq=tensor('nqc_auxiliary_orbital_frequency')
            grid_peak=aux[torch.argmax(freq)]; index=matching_index(aux,grid_peak-1.)
            selected=aux[index]; actual_peak=tensor('event_peak_time')
            p=run['case']['params']; nu=p['mass1']*p['mass2']/(p['mass1']+p['mass2'])**2
            amp=tensor('nqc_original_amplitude')/(nu*24.**.5)
            splines=[natural_cubic_spline(time,amp),natural_cubic_spline(time,amp[:,None]*tensor('nqc_original_basis_amplitude'),scalar_output=False),natural_cubic_spline(time,phase),natural_cubic_spline(time,tensor('nqc_original_basis_phase'),scalar_output=False)]
            a,aq,ph,b=splines
            targets=[tensor('nqc_target_'+key) for key in ('amplitude','amplitude_derivative','frequency','frequency_derivative')]
            variants={}
            for label,t in [('selected_grid_knot',selected),('grid_peak_minus_one',grid_peak-1.),('continuous_peak_minus_one',actual_peak-1.)]:
                am=torch.stack((aq(t),aq.derivative(t))); ar=torch.stack((targets[0]-a(t),targets[1]-a.derivative(t)))
                bm=torch.stack((b.derivative(t),b.derivative(t,2))); br=torch.stack((ph.derivative(t)-targets[2],ph.derivative(t,2)-targets[3]))
                coeff=[torch.linalg.solve(am,ar),torch.linalg.solve(bm,br)]
                variants[label]={'time_M':t,'time_before_continuous_target_M':actual_peak-1.-t,'coefficients':coeff,'phase_coefficient_distance_from_grid':torch.linalg.vector_norm(coeff[1]-torch.tensor(prior['nqc_coefficients'][1],dtype=torch.float64))}
            grid_coeff=torch.tensor(prior['nqc_coefficients'][1],dtype=torch.float64); continuous_coeff=tensor('nqc_phase_coefficients')
            report[name]={'cache_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'grid_coefficients':prior['nqc_coefficients'],'continuous_coefficients':run['native']['nqc_coefficients'],'selected_index':index,'continuous_spline_at':variants,'fraction_of_phase_coefficient_gap_remaining_at_grid_time':float(variants['selected_grid_knot']['phase_coefficient_distance_from_grid']/torch.linalg.vector_norm(continuous_coeff-grid_coeff.double()))}
    output=OUT/'nqc-time-controls.json';output.write_text(json.dumps(json_value(report),indent=2)+'\n')
    for name,d in report.items():print(name, 'gap remaining at grid time',d['fraction_of_phase_coefficient_gap_remaining_at_grid_time'])

if __name__=='__main__':main()
