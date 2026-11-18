"""Verificações sintéticas da análise por pixel e bootstrap, sem GPU ou rede."""
import csv
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src import analise_erros as analise


class TestAnalise(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/tmp", prefix="analise-sintetica-")
        self.addCleanup(self.temp.cleanup)
        self.raiz = Path(self.temp.name)

    @staticmethod
    def mascaras(classes):
        mapa = np.array(classes).reshape(2, 3)
        return np.array([mapa == c for c in (1, 2, 3)])

    def test_matriz_conhecida(self):
        gt = self.mascaras([0, 1, 2, 3, 1, 2])
        pred = self.mascaras([0, 1, 3, 2, 0, 1])
        cm, amb, total = analise.matriz_pixel(gt, pred)
        np.testing.assert_array_equal(cm, [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
        self.assertEqual((amb, total), (0, 6))

    def test_sobreposicao_em_gt_ou_pred_vira_ambiguo(self):
        gt = self.mascaras([0, 1, 2, 3, 1, 2])
        pred = gt.copy()
        gt[1, 0, 1] = True
        pred[2, 0, 2] = True
        cm, amb, total = analise.matriz_pixel(gt, pred)
        self.assertEqual((amb, total, cm.sum()), (2, 6, 4))
        self.assertEqual(cm[1, 1], 1)
        self.assertEqual(cm[2, 2], 1)

    def test_sem_objetos(self):
        gt = np.zeros((3, 2, 3), dtype=bool)
        pred = gt.copy()
        pred[0, 0, 0] = True
        cm, amb, total = analise.matriz_pixel(gt, pred)
        self.assertEqual(cm[0, 0], 5)
        self.assertEqual(cm[0, 1], 1)
        self.assertEqual((amb, total), (0, 6))
        i, p, g = analise.contagens(gt, pred)
        self.assertEqual((i.sum(), p.sum(), g.sum()), (0, 1, 0))

    def test_intersecao_iou_dice_conferidos_a_mao(self):
        gt = np.array([[[1, 1, 0]], [[0, 0, 0]], [[0, 0, 0]]])
        pred = np.array([[[1, 0, 1]], [[0, 0, 0]], [[0, 0, 0]]])
        inter, previsto, rotulo = analise.contagens(gt, pred)
        np.testing.assert_array_equal(inter, [1, 0, 0])
        np.testing.assert_array_equal(previsto, [2, 0, 0])
        np.testing.assert_array_equal(rotulo, [2, 0, 0])
        with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=Mock())}), patch('dotenv.load_dotenv'):
            from src.evaluate import calcular_metricas
        m = calcular_metricas(inter, previsto, rotulo, analise.CLASSES)
        self.assertAlmostEqual(m['construcao']['iou'], 1 / 3)
        self.assertAlmostEqual(m['construcao']['dice'], 1 / 2)
        self.assertNotIn('vegetacao', m)

    def test_bootstrap_reprodutivel_e_pareado(self):
        a = np.array([[[1, 2, 3], [2, 3, 4], [3, 5, 6]], [[2, 3, 4], [1, 2, 3], [4, 6, 7]]])
        b = a.copy()
        b[..., 0] += 1
        r = analise.bootstrap_pareado(a, b, 100, 42)
        self.assertEqual(r, analise.bootstrap_pareado(a, b, 100, 42))
        self.assertEqual(r['metricas']['mIoU']['diferenca_pp']['positivas'], 100)
        self.assertGreater(r['metricas']['mIoU']['diferenca_pp']['ic95'][0], 0)
        self.assertAlmostEqual(r['metricas']['mIoU']['A']['valor'], np.mean([3/9, 3/9, 7/17]))
        iguais = analise.bootstrap_pareado(a, a, 100, 42)
        self.assertEqual(iguais['metricas']['mIoU']['diferenca_pp']['ic95'], [0, 0])

    def test_bootstrap_rejeita_referencias_distintas(self):
        a = np.ones((2, 3, 3), dtype=int)
        b = a.copy()
        b[0, 0, 2] += 1
        with self.assertRaisesRegex(ValueError, 'Rótulos diferem'):
            analise.bootstrap_pareado(a, b)

    def escrever_contagens(self, pasta, nome):
        pasta.mkdir()
        with (pasta / 'por_imagem.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['imagem', 'classe', 'intersecao', 'previsao', 'rotulo'])
            w.writeheader()
            for classe in analise.CLASSES.values():
                w.writerow(dict(imagem=nome, classe=classe, intersecao=1, previsao=2, rotulo=3))

    def test_comparar_aborta_listas_distintas(self):
        a, b = self.raiz / 'a', self.raiz / 'b'
        self.escrever_contagens(a, 'a.jpg')
        self.escrever_contagens(b, 'b.jpg')
        with self.assertRaisesRegex(ValueError, 'Listas de imagens diferentes'):
            analise.comparar(a, b, self.raiz / 'saida')
        self.assertFalse((self.raiz / 'saida').exists())

    def test_comparar_grava_ic_sem_artefatos_de_modelo(self):
        a, b = self.raiz / 'a', self.raiz / 'b'
        self.escrever_contagens(a, 'a.jpg')
        self.escrever_contagens(b, 'a.jpg')
        saida = self.raiz / 'saida'
        r = analise.comparar(a, b, saida, 50, 42)
        self.assertEqual(json.loads((saida / 'comparacao.json').read_text()), r)
        self.assertEqual(r['metricas']['mDice']['diferenca_pp']['valor'], 0)

    def test_anonimizacao_e_avaliacao_integrada_com_stub(self):
        import cv2
        imagens, rotulos = self.raiz / 'imagens', self.raiz / 'rotulos'
        imagens.mkdir(); rotulos.mkdir()
        cv2.imwrite(str(imagens / 'foto.jpg'), np.zeros((16, 16, 3), np.uint8))
        (rotulos / 'foto.txt').write_text('0 0 0 1 0 1 1 0 1\n')
        pesos = self.raiz / 'modelo.pt'
        pesos.write_bytes(b'stub')
        grupos = self.raiz / 'sanidade_divisao.json'
        grupos.write_text(json.dumps({'imagens': [{'nome': 'foto.jpg', 'local': 'LOCAL PRIVADO'}]}))
        interno = self.raiz / 'scripts/mapa_locais_INTERNO.csv'
        modelo = Mock()
        with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=modelo)}), patch('dotenv.load_dotenv'):
            evaluate = importlib.import_module('src.evaluate')
            pred = np.zeros((3, 16, 16), dtype=np.uint8)
            pred[0] = 1
            with patch.object(evaluate, 'prever_mascaras', return_value=pred):
                r = analise.avaliar(pesos, imagens, rotulos, self.raiz / 'saida', 640, .25, grupos, interno)
        self.assertEqual(r['mIoU'], 1)
        self.assertEqual(r['imagens'], 1)
        self.assertEqual(list(r['locais']), ['local_A'])
        self.assertIn('LOCAL PRIVADO', interno.read_text())
        for arquivo in (self.raiz / 'saida').iterdir():
            self.assertNotIn('LOCAL PRIVADO', arquivo.read_text())
        self.assertEqual(r['confusao_pixel']['contagens'][1][1], 256)
        self.assertEqual(r['por_classe']['construcao']['precisao'], 1)
        modelo.assert_called_once_with(str(pesos))

    def test_grupos_le_selecao_adjacente(self):
        grupos = self.raiz / 'sanidade_divisao.json'
        grupos.write_text('{"contagens": {"val": 1}}')
        (self.raiz / 'selecao.csv').write_text('nome;local\nfoto.jpg;PRIVADO\n')
        m = analise.carregar_grupos(grupos, self.raiz / 'scripts/mapa_locais_INTERNO.csv')
        self.assertEqual(m, {'foto.jpg': 'local_A'})


if __name__ == '__main__':
    unittest.main()
