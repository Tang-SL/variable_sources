-- ============================================================
-- HSC-SSP PDR3  |  Step 1: Find object_ids + coadd photometry
-- Schema: pdr3_dud  (Deep+UltraDeep, covers COSMOS)
--
-- boxSearch(coord, ra1, ra2, dec1, dec2) — all in DEGREES
-- Each box is ±1 arcsec (0.000278 deg) around each CID position.
--
-- Paste in CAS at: https://hsc-release.mtk.nao.ac.jp/datasearch/
-- ============================================================

SELECT
  f.object_id,
  f.ra,
  f.dec,
  f.g_cmodel_mag,
  f.g_cmodel_magerr,
  f.r_cmodel_mag,
  f.r_cmodel_magerr,
  f.i_cmodel_mag,
  f.i_cmodel_magerr,
  f.z_cmodel_mag,
  f.z_cmodel_magerr,
  f.y_cmodel_mag,
  f.y_cmodel_magerr
FROM pdr3_dud.forced AS f
WHERE (
  boxSearch(coord, 150.179522, 150.180078, 2.110062, 2.110618)   -- cid_42
  OR boxSearch(coord, 150.087163, 150.087719, 1.741656, 1.742212)  -- cid_268
  OR boxSearch(coord, 149.930592, 149.931148, 2.118559, 2.119115)  -- cid_346
  OR boxSearch(coord, 150.004102, 150.004658, 2.038700, 2.039256)  -- cid_349
  OR boxSearch(coord, 150.002292, 150.002848, 2.258385, 2.258941)  -- cid_451
  OR boxSearch(coord, 149.920262, 149.920818, 2.543390, 2.543946)  -- cid_563
  OR boxSearch(coord, 150.010412, 150.010968, 2.332723, 2.333279)  -- cid_1205
  OR boxSearch(coord, 149.832314, 149.832870, 2.710581, 2.711137)  -- cid_1605
  OR boxSearch(coord, 149.874252, 149.874808, 2.361211, 2.361767)  -- cid_2550
)
AND isprimary;
