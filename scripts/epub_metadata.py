"""Declare MathML in all EPUB content documents, including the navigation TOC."""
import os
import posixpath
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile


def declare_mathml(epub_path):
    with ZipFile(epub_path) as archive:
        container = ET.fromstring(archive.read('META-INF/container.xml'))
        opf_path = container.find('.//{*}rootfile').get('full-path')
        opf = ET.fromstring(archive.read(opf_path))
        changed = False
        for item in opf.findall('.//{*}manifest/{*}item'):
            if item.get('media-type') != 'application/xhtml+xml':
                continue
            name = posixpath.normpath(posixpath.join(posixpath.dirname(opf_path), item.get('href')))
            content = ET.fromstring(archive.read(name))
            if content.find('.//{http://www.w3.org/1998/Math/MathML}math') is not None:
                properties = set(item.get('properties', '').split())
                if 'mathml' not in properties:
                    properties.add('mathml')
                    item.set('properties', ' '.join(sorted(properties)))
                    changed = True
        if not changed:
            return
        ET.register_namespace('', 'http://www.idpf.org/2007/opf')
        ET.register_namespace('dc', 'http://purl.org/dc/elements/1.1/')
        updated = ET.tostring(opf, encoding='utf-8', xml_declaration=True)
        fd, temporary = tempfile.mkstemp(suffix='.epub', dir=os.path.dirname(epub_path))
        os.close(fd)
        try:
            with ZipFile(temporary, 'w') as output:
                for entry in archive.infolist():
                    output.writestr(entry, updated if entry.filename == opf_path else archive.read(entry.filename))
            os.replace(temporary, epub_path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
